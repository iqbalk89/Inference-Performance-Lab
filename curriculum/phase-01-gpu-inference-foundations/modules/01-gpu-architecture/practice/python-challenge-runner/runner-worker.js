let pyodidePromise;
async function getPyodide() {
  if (!pyodidePromise) {
    importScripts("https://cdn.jsdelivr.net/pyodide/v0.29.2/full/pyodide.js");
    pyodidePromise = loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.2/full/" });
  }
  return pyodidePromise;
}
self.onmessage = async function (event) {
  const request = event.data;
  if (request.type === "initialize") {
    try { await getPyodide(); self.postMessage({ type: "ready" }); }
    catch (error) { self.postMessage({ type: "fatal", message: String(error) }); }
    return;
  }
  if (request.type !== "run") return;
  try {
    const pyodide = await getPyodide();
    pyodide.globals.set("_user_code", request.code);
    pyodide.globals.set("_function_name", request.functionName);
    pyodide.globals.set("_tests_json", JSON.stringify(request.tests));
    const resultJson = await pyodide.runPythonAsync([
      "import contextlib, io, json, traceback",
      "_tests = json.loads(_tests_json)",
      "_namespace, _results = {}, []",
      "_stdout, _compile_error = io.StringIO(), None",
      "try:",
      "    with contextlib.redirect_stdout(_stdout):",
      "        exec(compile(_user_code, '<student solution>', 'exec'), _namespace)",
      "except BaseException:",
      "    _compile_error = traceback.format_exc()",
      "if _compile_error is None and _function_name not in _namespace:",
      "    _compile_error = f'Required function {_function_name} was not defined.'",
      "if _compile_error is None:",
      "    _function = _namespace[_function_name]",
      "    for _index, _test in enumerate(_tests):",
      "        _record = {'index': _index, 'hidden': _test.get('hidden', False), 'label': _test.get('label', f'test {_index + 1}'), 'args': _test.get('args', []), 'expected': _test.get('expected'), 'expectedException': _test.get('expectedException')}",
      "        try:",
      "            with contextlib.redirect_stdout(_stdout):",
      "                _actual = _function(*_test.get('args', []))",
      "            _record['actual'] = _actual",
      "            if _test.get('expectedException'):",
      "                _record['passed'], _record['message'] = False, 'Expected ' + _test['expectedException'] + ' but no exception was raised.'",
      "            else:",
      "                _record['passed'] = _actual == _test.get('expected')",
      "                _record['message'] = '' if _record['passed'] else 'Returned value did not match expected value.'",
      "        except BaseException as _error:",
      "            _actual_exception = type(_error).__name__",
      "            _record['actualException'] = _actual_exception",
      "            _record['passed'] = _actual_exception == _test.get('expectedException')",
      "            _record['message'] = str(_error) if _record['passed'] else traceback.format_exc()",
      "        _results.append(_record)",
      "json.dumps({'compileError': _compile_error, 'stdout': _stdout.getvalue(), 'results': _results}, default=repr)",
    ].join("\n"));
    self.postMessage({ type: "result", payload: JSON.parse(resultJson) });
  } catch (error) {
    self.postMessage({ type: "fatal", message: String(error) });
  }
};
