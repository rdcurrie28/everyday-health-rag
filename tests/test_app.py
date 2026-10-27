import dataclasses

import anthropic
import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from healthrag import config, retrieval
from healthrag.config import REPO_ROOT

from .conftest import done_reply, tool_reply

APP = str(REPO_ROOT / "app.py")
REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.fixture
def app(mini_index, fake_chat, monkeypatch):
    settings, _ = mini_index

    def make(**overrides):
        patched = dataclasses.replace(settings, **overrides)
        monkeypatch.setattr(config, "get_settings", lambda: patched)
        st.cache_resource.clear()
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        return at

    return make


def chat_text(at) -> str:
    return " ".join(md.value for msg in at.chat_message for md in msg.markdown)


def test_example_question_then_follow_up(app, fake_chat):
    at = app()
    assert not at.exception
    at.button(key="example-0").click().run()
    assert not at.exception
    assert "Hello world" in chat_text(at)
    assert at.session_state.turn_count == 1
    fake_chat.responses = [tool_reply("children physical activity"), done_reply()]
    at.button(key="follow-up").click().run()
    assert at.session_state.turn_count == 2
    assert len(at.session_state.turns) == 2
    assert any("children physical activity" in c.value for c in at.caption)


def chips_html(at) -> str:
    return " ".join(str(el.proto) for el in at.get("html"))


def test_chips_show_only_cited_pages(app, fake_chat):
    fake_chat.stream_pieces = ["Swim laps ", "(Mini Guide, p. ii)."]
    at = app()
    at.chat_input[0].set_value("Tell me about swimming").run()
    html = chips_html(at)
    assert "Cited:" in html and "Mini Guide · p. ii" in html
    assert "PDF p. 1" not in html


def test_no_chips_without_citations(app):
    at = app()
    at.chat_input[0].set_value("Tell me about swimming").run()
    assert "Cited:" not in chips_html(at)


def test_off_topic_question_is_not_covered(app, fake_chat, monkeypatch):
    fake_chat.responses = [tool_reply("capital of France"), done_reply()]
    monkeypatch.setattr(retrieval.Retriever, "max_distance", property(lambda self: -1.0))
    at = app()
    at.chat_input[0].set_value("What is the capital of France?").run()
    assert not at.exception
    assert "don't seem to cover this" in chat_text(at)
    assert fake_chat.streams == []
    assert len(at.session_state.turns) == 1


def test_typed_question_and_new_chat(app):
    at = app()
    at.chat_input[0].set_value("Tell me about swimming").run()
    assert len(at.session_state.turns) == 1
    new_chat = next(b for b in at.button if b.label == "New chat")
    new_chat.click().run()
    assert at.session_state.turns == []
    assert at.session_state.turn_count == 1


def test_llm_error_shows_message_and_skips_turn(app, fake_chat):
    fake_chat.stream_error = anthropic.APIConnectionError(request=REQUEST)
    at = app()
    at.chat_input[0].set_value("Tell me about swimming").run()
    assert any("Busy right now" in e.value for e in at.error)
    assert at.session_state.turns == []
    assert at.session_state.turn_count == 1


def test_session_cap(app):
    at = app()
    at.session_state.turn_count = config.MAX_TURNS_PER_SESSION
    at.chat_input[0].set_value("One more?").run()
    assert any("Session limit reached" in w.value for w in at.warning)


def test_password_gate(app):
    at = app(app_password="open sesame")
    assert not at.chat_input
    at.text_input[0].input("wrong")
    at.button[0].click().run()
    assert any("Wrong password" in e.value for e in at.error)
    at.text_input[0].input("open sesame")
    at.button[0].click().run()
    assert at.session_state.authed
    assert at.chat_input


def test_missing_index_shows_error(app, tmp_path):
    at = app(index_dir=tmp_path / "missing")
    assert any("Search index not found" in e.value for e in at.error)


def test_deployed_without_secrets(app):
    at = app(app_mode="deployed", anthropic_api_key=None, app_password=None)
    assert any("Demo not configured" in e.value for e in at.error)
