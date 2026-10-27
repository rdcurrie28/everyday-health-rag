import dataclasses

from healthrag.config import check_password, get_settings


def test_check_password():
    assert check_password("open sesame", "open sesame")
    assert not check_password("open", "open sesame")
    assert check_password("pässwörd✓", "pässwörd✓")
    assert not check_password("pässwörd", "password")


def test_missing_secrets():
    settings = dataclasses.replace(get_settings(), anthropic_api_key=None, app_password=None)
    assert settings.missing_secrets() == ["ANTHROPIC_API_KEY", "APP_PASSWORD"]
    settings = dataclasses.replace(settings, anthropic_api_key="k", app_password="p")
    assert settings.missing_secrets() == []


def test_secrets_not_in_repr():
    settings = dataclasses.replace(
        get_settings(), anthropic_api_key="secret-key", app_password="pw"
    )
    assert "secret-key" not in repr(settings) and "pw'" not in repr(settings)
