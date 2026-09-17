import py_compile
import sys
import traceback

try:
    py_compile.compile('aidas/app.py', doraise=True)
    print("app.py OK")
except Exception as e:
    print("app.py ERROR:", e)

try:
    py_compile.compile('aidas/ui/components.py', doraise=True)
    print("components.py OK")
except Exception as e:
    print("components.py ERROR:", e)

