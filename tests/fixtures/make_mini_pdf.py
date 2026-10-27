from pathlib import Path

PAGE_TEXT = [
    [
        "Mini guide page one covers fruit and vegetables for the test suite.",
        "Oranges and kiwis are bright fruits that supply vitamin C every day.",
        "Carrots, spinach, and peppers add color and fiber to a healthy plate.",
    ],
    [
        "Mini guide page two covers swimming and other water activities.",
        "Swimming laps for thirty minutes is a moderate aerobic activity.",
        "Water aerobics classes help older adults build strength and balance.",
    ],
]
REPEATS = 7


def _stream(lines: list[str]) -> bytes:
    body = ["BT", "/F1 9 Tf", "11 TL", "40 760 Td"]
    for n in range(REPEATS):
        for line in lines:
            body.append(f"({line} [{n + 1}]) Tj T*")
    body.append("ET")
    return "\n".join(body).encode("latin-1")


def build_pdf() -> bytes:
    streams = [_stream(lines) for lines in PAGE_TEXT]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /PageLabels << /Nums [0 << /S /D >> 1 << /P (ii) >>] >> >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(streams[0]) + streams[0] + b"\nendstream",
        b"<< /Length %d >>\nstream\n" % len(streams[1]) + streams[1] + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


if __name__ == "__main__":
    target = Path(__file__).with_name("mini.pdf")
    target.write_bytes(build_pdf())
    print(f"wrote {target}")
