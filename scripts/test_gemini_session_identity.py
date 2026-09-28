"""Verify Gemini route identity against an actual executable, without inference.

Usage:
    JCODE_SCRATCH_DIR=/short/scratch python3 scripts/test_gemini_session_identity.py /path/to/jcode

Uses private HOME/runtime/socket, fake OAuth and an auth-free local profile.
Never inherits real credentials, submits inference, or reloads a shared daemon.
Leaves logs/results in scratch and always terminates its own daemon processes.
"""
import json, os, socket, subprocess, sys, tempfile, time
from pathlib import Path

BINARY = str(Path(sys.argv[1]).resolve())
ROOT = Path(tempfile.mkdtemp(prefix='gi-', dir=os.environ['JCODE_SCRATCH_DIR']))
HOME = ROOT / 'home'
STATE = HOME / '.jcode'
WORK = HOME / 'work'
RUNTIME = ROOT / 'rt'
for p in (STATE, WORK, RUNTIME):
    p.mkdir(parents=True, exist_ok=True)
SOCKET = str(ROOT / 's.sock')
(STATE / 'gemini_oauth.json').write_text(json.dumps({
    'access_token': 'isolated-not-real-oauth', 'refresh_token': 'unused',
    'expires_at': int(time.time() * 1000) + 3600000,
}))
(STATE / 'config.toml').write_text('''
[providers.fixture]
type = "openai-compatible"
base_url = "http://127.0.0.1:1/v1"
requires_api_key = false
auth = "none"
default_model = "fixture-model"
model_catalog = false
''')
ENV = {
    'PATH': '/usr/bin:/bin:/opt/homebrew/bin', 'HOME': str(HOME),
    'JCODE_HOME': str(STATE), 'JCODE_RUNTIME_DIR': str(RUNTIME),
    'JCODE_SCRATCH_DIR': str(ROOT), 'TMPDIR': str(RUNTIME),
    'JCODE_GEMINI_FORCE_OAUTH': '1',
    'CODE_ASSIST_ENDPOINT': 'http://127.0.0.1:1',
    'JCODE_RUNTIME_PROVIDER': 'openai-compatible',
    'JCODE_OPENROUTER_CACHE_NAMESPACE': 'fixture',
    'HTTP_PROXY': 'http://127.0.0.1:1', 'HTTPS_PROXY': 'http://127.0.0.1:1',
    'ALL_PROXY': 'http://127.0.0.1:1', 'NO_PROXY': '127.0.0.1,localhost',
    'TERM': 'dumb', 'JCODE_NO_TELEMETRY': '1',
}
proc = None
client = None
log = None
phase = 0
reports = []

class Client:
    def __init__(self):
        self.s = socket.socket(socket.AF_UNIX)
        self.s.settimeout(20)
        self.s.connect(SOCKET)
        self.f = self.s.makefile('rb')
        self.seq = 0
    def send(self, kind, **fields):
        self.seq += 1
        msg = dict(type=kind, id=self.seq, **fields)
        self.s.sendall((json.dumps(msg) + '\n').encode())
        return self.seq
    def until(self, kind, request_id=None):
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            line = self.f.readline()
            assert line, 'server connection closed'
            msg = json.loads(line)
            with (ROOT / 'events.jsonl').open('a') as f:
                f.write(json.dumps(msg) + '\n')
            if msg.get('type') == 'error':
                raise AssertionError(msg)
            if msg.get('type') == kind and (request_id is None or msg.get('id') == request_id):
                return msg
        raise AssertionError('timeout waiting for ' + kind)
    def history(self):
        return self.until('history', self.send('get_history'))
    def model(self, model):
        result = self.until('model_changed', self.send('set_model', model=model))
        assert not result.get('error'), result
        return result
    def route(self, model, runtime, api, label):
        result = self.until('model_changed', self.send('set_route', selection={
            'model': model, 'runtime_key': runtime, 'api_method': api,
            'provider_label': label, 'detail': 'isolated regression fixture',
        }))
        assert not result.get('error'), result
        return result
    def close(self):
        self.f.close()
        self.s.close()


def start(provider='gemini', sid=None):
    global proc, client, log, phase
    phase += 1
    log = (ROOT / ('daemon-%d.log' % phase)).open('w')
    args = [BINARY, '--no-update', '--no-selfdev', '--socket', SOCKET,
            '-C', str(WORK), '--tool-profile', 'none']
    if provider == 'gemini':
        args += ['--provider', 'gemini', '--model', 'gemini-pro-latest']
    elif provider == 'auto':
        args += ['--provider', 'auto', '--model', 'gemini:gemini-pro-latest']
    else:
        args += ['--provider', 'auto', '--model', 'fixture:fixture-model']
    args += ['serve', '--temporary-server', '--owner-pid', str(os.getpid()),
             '--temp-idle-timeout-secs', '120', '--server-name', 'isolated-gemini-check']
    proc = subprocess.Popen(args, env=ENV, cwd=WORK, stdout=log, stderr=log)
    deadline = time.monotonic() + 25
    while not Path(SOCKET).exists():
        assert proc.poll() is None, (proc.returncode, (ROOT / ('daemon-%d.log' % phase)).read_text())
        assert time.monotonic() < deadline, 'socket did not appear'
        time.sleep(.05)
    client = Client()
    fields = {'working_dir': str(WORK), 'client_instance_id': 'isolated-fixture-%d' % phase}
    if sid:
        fields['target_session_id'] = sid
    client.send('subscribe', **fields)
    assigned = client.until('session')['session_id']
    if sid:
        assert assigned == sid, (assigned, sid)
    return assigned


def stop():
    global client, proc, log
    if client:
        client.close()
        client = None
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=12)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        proc = None
    if log:
        log.close()
        log = None
    Path(SOCKET).unlink(missing_ok=True)


def saved(sid):
    data = json.loads((STATE / 'sessions' / (sid + '.json')).read_text())
    journal = STATE / 'sessions' / (sid + '.journal.jsonl')
    if journal.exists():
        for line in journal.read_text().splitlines():
            entry = json.loads(line)
            data.update(entry['meta'])
            for vector in ('messages', 'env_snapshots', 'memory_injections', 'replay_events'):
                data.setdefault(vector, []).extend(entry.get('append_' + vector, []))
    return data


def check(label, sid, marker=True, route=None):
    history = client.history()
    assert history['session_id'] == sid, history
    assert history['provider_name'].casefold() == 'gemini', history.get('provider_name')
    assert history['provider_model'] == 'gemini-pro-latest', history.get('provider_model')
    if marker:
        assert 'ISOLATED_HISTORY_MARKER' in json.dumps(history['messages']), history['messages']
    state = client.until('state', client.send('state'))
    assert not state['is_processing'], state
    data = saved(sid)
    expected_key = 'code-assist-oauth' if route == 'code-assist-oauth' else 'gemini'
    assert data['provider_key'] == expected_key, data.get('provider_key')
    if route is not None:
        assert data.get('route_api_method') == route, data.get('route_api_method')
    record = {'case': label, 'session_id': sid, 'provider': history['provider_name'],
              'model': history['provider_model'], 'provider_key': data['provider_key'],
              'route_api_method': data.get('route_api_method'), 'messages': len(history['messages']),
              'processing': state['is_processing'], 'result': 'PASS'}
    reports.append(record)
    print(json.dumps(record), flush=True)

print('Evidence:', ROOT, flush=True)
try:
    sid = start()
    mid = client.send('message', content='ISOLATED_HISTORY_MARKER', no_reply=True)
    # Requests are processed serially. Reading history waits past the no-reply append.
    check('create_native_session_with_ambient_compatible_env', sid)
    stop()
    start(provider='auto', sid=sid)
    check('auto_daemon_restores_native_session', sid)
    client.route('fixture-model', {'kind': 'open-ai-compatible', 'profile_id': 'fixture'},
                 'openai-compatible:fixture', 'fixture')
    h = client.history()
    assert h['provider_name'].casefold() != 'gemini', h.get('provider_name')
    assert h['provider_model'] == 'fixture-model', h.get('provider_model')
    client.route('gemini-pro-latest', {'kind': 'code-assist-o-auth'}, 'code-assist-oauth', 'gemini')
    check('typed_picker_exits_compatible_endpoint', sid, route='code-assist-oauth')
    stop()
    start(provider='fixture', sid=sid)
    check('restart_preserves_oauth_identity_and_history', sid, route='code-assist-oauth')
    client.route('fixture-model', {'kind': 'open-ai-compatible', 'profile_id': 'fixture'},
                 'openai-compatible:fixture', 'fixture')
    client.model('gemini:gemini-pro-latest')
    check('explicit_model_switch_clears_stale_typed_route', sid)
    assert saved(sid).get('route_api_method') is None, saved(sid).get('route_api_method')
    stop()
    start(provider='fixture', sid=sid)
    check('explicit_switch_survives_restart', sid)
    stop()
    # Reproduce old incorrect provider key with an explicit OAuth route in isolated data only.
    data = saved(sid)
    data['provider_key'] = 'openai-compatible'
    data['route_api_method'] = 'code-assist-oauth'
    (STATE / 'sessions' / (sid + '.json')).write_text(json.dumps(data))
    (STATE / 'sessions' / (sid + '.journal.jsonl')).unlink(missing_ok=True)
    start(provider='fixture', sid=sid)
    h = client.history()
    assert h['provider_name'].casefold() == 'gemini', h.get('provider_name')
    assert h['provider_model'] == 'gemini-pro-latest', h.get('provider_model')
    assert 'ISOLATED_HISTORY_MARKER' in json.dumps(h['messages'])
    reports.append({'case': 'typed_oauth_restores_despite_legacy_wrong_provider_key', 'result': 'PASS'})
    print(json.dumps(reports[-1]), flush=True)
finally:
    stop()
    (ROOT / 'results.json').write_text(json.dumps(reports, indent=2))
print('PASS: private daemon lifecycle only. No model turn requested. Shared daemon and sessions untouched.', flush=True)
