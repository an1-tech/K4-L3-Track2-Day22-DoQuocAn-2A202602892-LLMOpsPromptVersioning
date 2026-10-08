"""Bước 4: custom PII và JSON validators."""

import argparse
import ast
import json
import re

from guardrails import Guard
from guardrails.validators import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
)
from guardrails.types import OnFailAction


@register_validator(
    name="custom/pii-detector",
    data_type="string",
)
class PIIDetector(Validator):
    """Phát hiện bốn dạng PII của lab bằng regex."""

    PII_PATTERNS = {
        "EMAIL": (
            r"\b[A-Za-z0-9._%+-]+@"
            r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        "CREDIT_CARD": (
            r"(?<!\d)(?:\d{4}[- ]?){3}\d{4}(?!\d)"
        ),
        "SSN": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
        "PHONE": (
            r"(?<!\w)(?:\+?1[-.\s]?)?"
            r"(?:\(\d{3}\)|\d{3})"
            r"[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
        ),
    }

    def validate(self, value: str, metadata: dict):
        redacted = value
        found = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, redacted):
                found.append(pii_type)

                redacted = re.sub(
                    pattern,
                    f"[{pii_type}_REDACTED]",
                    redacted,
                )

        if found:
            return FailResult(
                error_message=f"PII: {', '.join(found)}",
                fix_value=redacted,
            )

        return PassResult()


@register_validator(
    name="custom/json-formatter",
    data_type="string",
)
class JSONFormatter(Validator):
    """Sửa fences, nháy đơn và trailing commas."""

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(r"\s*```$", "", text).strip()

        # Sau khi bỏ fences, thử JSON trước.
        try:
            return json.dumps(
                json.loads(text),
                ensure_ascii=False,
            )
        except json.JSONDecodeError:
            pass

        # Hỗ trợ dict nháy đơn; không thực thi code.
        try:
            parsed = ast.literal_eval(text)

            return json.dumps(
                parsed,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (ValueError, SyntaxError, TypeError):
            pass

        # Xóa trailing comma ngoài chuỗi.
        # Giữ nguyên ví dụ: {"text": "hello,}"}
        text = re.sub(
            r'"(?:\\.|[^"\\])*"|,\s*(?=[}\]])',
            lambda m: (
                m.group(0)
                if m.group(0).startswith('"')
                else ""
            ),
            text,
        )

        return text

    def validate(self, value: str, metadata: dict):
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        try:
            parsed = json.loads(self._repair(value))

            fixed = json.dumps(
                parsed,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

            return FailResult(
                error_message="JSON lỗi, đã tự sửa",
                fix_value=fixed,
            )

        except (json.JSONDecodeError, ValueError):
            fallback = json.dumps(
                {"error": "Không thể phân tích JSON"},
                ensure_ascii=False,
            )

            return FailResult(
                error_message="Không thể sửa JSON; dùng fallback",
                fix_value=fallback,
            )


def demo_pii_guard():
    guard = Guard().use(
        PIIDetector(on_fail=OnFailAction.FIX)
    )

    test_cases = [
        (
            "Email",
            "Contact John at john.doe@example.com for details.",
            "[EMAIL_REDACTED]",
        ),
        (
            "Phone",
            "Call our support line at (555) 867-5309.",
            "[PHONE_REDACTED]",
        ),
        (
            "SSN",
            "Patient SSN is 123-45-6789 on file.",
            "[SSN_REDACTED]",
        ),
        (
            "Credit Card",
            "Payment made with card 4532 1234 5678 9010.",
            "[CREDIT_CARD_REDACTED]",
        ),
        (
            "Multi-PII",
            "Email: alice@example.com, Phone: 555-123-4567",
            "[EMAIL_REDACTED]",
        ),
        (
            "Clean",
            "No sensitive information in this text.",
            None,
        ),
    ]

    print("=== PII Detection & Redaction: 6 cases ===")

    for label, text, marker in test_cases:
        result = guard.validate(text)
        output = result.validated_output

        assert isinstance(output, str)

        if marker:
            assert marker in output
        else:
            assert output == text

        if label == "Multi-PII":
            assert "[PHONE_REDACTED]" in output

        assert not any(
            re.search(pattern, output)
            for pattern in PIIDetector.PII_PATTERNS.values()
        )

        print(f"\n[{label}] Kiểm tra output: OK")
        print(f"Input: {text}")
        print(f"Output: {output}")


def demo_json_guard():
    guard = Guard().use(
        JSONFormatter(on_fail=OnFailAction.FIX)
    )

    test_cases = [
        (
            "Valid JSON",
            '{"name": "Alice", "age": 30}',
            {"name": "Alice", "age": 30},
        ),
        (
            "Markdown fences",
            '```json\n{"name": "Bob"}\n```',
            {"name": "Bob"},
        ),
        (
            "Single quotes",
            "{'name': 'Charlie', 'score': 95}",
            {"name": "Charlie", "score": 95},
        ),
        (
            "Trailing comma",
            '{"key": "value",}',
            {"key": "value"},
        ),
        (
            "Truly invalid",
            "This is not JSON at all: ??? {]",
            {"error": "Không thể phân tích JSON"},
        ),
    ]

    print("=== JSON Formatting & Repair: 5 cases ===")

    for label, text, expected in test_cases:
        result = guard.validate(text)
        parsed = json.loads(result.validated_output)

        assert parsed == expected

        status = (
            "FALLBACK"
            if label == "Truly invalid"
            else "OK"
        )

        print(f"\n[{label}] {status}; Kiểm tra output: OK")
        print(f"Input: {text}")
        print(f"Output: {result.validated_output}")


def main(argv=()):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--demo",
        choices=["pii", "json", "all"],
        default="all",
    )

    args = parser.parse_args(argv)

    if args.demo in ("pii", "all"):
        demo_pii_guard()

    if args.demo in ("json", "all"):
        demo_json_guard()


if __name__ == "__main__":
    main(None)