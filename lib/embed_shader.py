#!/usr/bin/env python3
"""Wrap shader.comp in a raw string literal, for compilers without C23 #embed.

The output must declare exactly what the #embed path declares -- const char knlmGlsl[],
NUL terminated -- so knlmvulkan.cpp compiles identically through either route. The file is
only rewritten when its content changes, so an untouched shader never dirties the build.
"""
import sys

DELIM = "KNLMVKSHADER"


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        text = f.read()
    if f"){DELIM}\"" in text:
        print(f"{src}: contains the raw string delimiter ){DELIM}\", pick another",
              file=sys.stderr)
        return 1
    out = f'const char knlmGlsl[] = R"{DELIM}({text}){DELIM}";\n'
    try:
        with open(dst, encoding="utf-8", newline="") as f:
            if f.read() == out:
                return 0
    except OSError:
        pass
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
