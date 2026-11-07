"""Run AndroidWorld T3A with Gemma served by the USB-connected Android phone.

The AndroidWorld task environment is the official API-33 emulator on Windows.
Only the model executes on the physical phone. No cloud-model fallback exists.
"""
import argparse
import csv
import http.client
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class PhoneGemma:
    def __init__(self, base_url, model, log_path, max_tokens=512, timeout=600):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'}:
            raise ValueError('This guide requires the local adb-forward endpoint, http://127.0.0.1:8080/v1')
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.calls = 0
        self.call_durations = []  # seconds per call (including context-overflow calls), in order
        self.context_overflows = 0
        self.connection_retries = 0
        self.max_retries = 3
        self.phone_serial = None  # set to re-create `adb forward tcp:8080 tcp:8080` after a dropped connection
        self.adb_path = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _reforward(self):
        if not (self.phone_serial and self.adb_path):
            return
        port = urllib.parse.urlsplit(self.base_url).port or 80
        subprocess.run([self.adb_path, '-s', self.phone_serial, 'wait-for-device'], timeout=300, check=False)
        subprocess.run([self.adb_path, '-s', self.phone_serial, 'forward', f'tcp:{port}', f'tcp:{port}'],
                       timeout=30, check=False)

    def check_model(self):
        with self.opener.open(self.base_url + '/models', timeout=30) as response:
            models = json.load(response)
        ids = [item.get('id') for item in models.get('data', [])]
        if self.model not in ids:
            raise RuntimeError(f'Expected server alias {self.model!r}; found {ids}. Start start_phone.sh.')

    def predict(self, text_prompt):
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': text_prompt}],
            'temperature': 0,
            'seed': 42,
            'max_tokens': self.max_tokens,
            'stream': False,
            'cache_prompt': False,
        }
        request = urllib.request.Request(
            self.base_url + '/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )
        started = time.perf_counter()
        print(f'[PHONE GEMMA] request {self.calls + 1} -> {request.full_url}', flush=True)
        attempt = 0
        while True:
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode('utf-8', errors='replace')
                if error.code == 400 and 'exceed_context_size_error' in detail:
                    return self._context_overflow(request.full_url, text_prompt, detail, time.perf_counter() - started)
                raise RuntimeError(f'Phone llama-server HTTP {error.code}: {detail}') from error
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError) as error:
                reason = getattr(error, 'reason', error)
                read_timeout = isinstance(reason, TimeoutError) and getattr(reason, 'errno', None) is None
                if read_timeout or attempt >= self.max_retries:
                    raise
                attempt += 1
                self.connection_retries += 1
                print(f'[PHONE GEMMA] connection lost ({type(error).__name__}: {error}); '
                      f're-forwarding adb and retrying the same request ({attempt}/{self.max_retries})', flush=True)
                self._reforward()
                time.sleep(15)
        text = result['choices'][0]['message'].get('content')
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError('Empty model answer. Check start_phone.sh uses --reasoning-budget 0 and inspect the phone log.')
        self.calls += 1
        elapsed = time.perf_counter() - started
        self.call_durations.append(elapsed)
        record = {
            'call': self.calls,
            'endpoint': request.full_url,
            'elapsed_s': elapsed,
            'prompt': text_prompt,
            'response': result,
        }
        with self.log_path.open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
        return text, None, result

    def _context_overflow(self, endpoint, text_prompt, detail, elapsed):
        """Prompt longer than the phone server's context window.

        The on-device model cannot read this input, so the call yields no usable
        answer: T3A receives a non-action text, treats the step as an invalid
        output, and the episode continues (and is judged by AndroidWorld as usual)
        instead of being aborted as an exception. Every occurrence is logged and
        counted in summary.json as context_overflow_calls.
        """
        try:
            info = json.loads(detail).get('error', {})
        except ValueError:
            info = {}
        self.context_overflows += 1
        self.call_durations.append(elapsed)
        record = {
            'call': None,
            'endpoint': endpoint,
            'elapsed_s': elapsed,
            'error': 'context_overflow',
            'n_prompt_tokens': info.get('n_prompt_tokens'),
            'n_ctx': info.get('n_ctx'),
            'prompt': text_prompt,
        }
        with self.log_path.open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'[PHONE GEMMA] CONTEXT OVERFLOW: prompt {info.get("n_prompt_tokens")} tokens > n_ctx {info.get("n_ctx")}; '
              'step gets no model answer', flush=True)
        text = (f'CONTEXT_OVERFLOW: the input ({info.get("n_prompt_tokens")} tokens) exceeds the on-device '
                f'model context ({info.get("n_ctx")} tokens); no answer was produced.')
        return text, None, {'error': 'context_overflow', **info}


class StepTimer:
    """Wraps agent.reset/agent.step and appends one CSV row per GUI-agent step.

    Rows are written immediately, so an interrupted run keeps the steps done so far.
    T3A makes up to two model calls per step: action selection, then summary.
    """

    FIELDS = [
        'task_template', 'goal', 'episode', 'step', 'started_at', 'step_s',
        'llm_calls', 'action_llm_s', 'summary_llm_s', 'llm_total_s', 'non_llm_s',
        'done', 'action', 'error',
    ]

    def __init__(self, path, llm, goal_to_task):
        self.path = Path(path)
        self.llm = llm
        self.goal_to_task = goal_to_task
        self.episode = 0
        self.step_index = 0

    def wrap(self, agent):
        original_reset, original_step = agent.reset, agent.step

        def reset(*args, **kwargs):
            self.episode += 1
            self.step_index = 0
            return original_reset(*args, **kwargs)

        def step(goal):
            self.step_index += 1
            first_call = len(self.llm.call_durations)
            started_at = time.strftime('%Y-%m-%d %H:%M:%S')
            started = time.perf_counter()
            result, error = None, ''
            try:
                result = original_step(goal)
                return result
            except BaseException as exc:
                error = f'{type(exc).__name__}: {exc}'[:500]
                raise
            finally:
                self._write(goal, started_at, time.perf_counter() - started,
                            self.llm.call_durations[first_call:], result, error)

        agent.reset, agent.step = reset, step

    def _write(self, goal, started_at, step_s, durations, result, error):
        data = getattr(result, 'data', None) or {}
        action = data.get('action_output') or ''
        for line in str(action).splitlines():
            if line.strip().startswith('Action:'):
                action = line.strip()[len('Action:'):].strip()
                break
        else:
            action = ' '.join(str(action).split())[:300]
        llm_total = sum(durations)
        row = {
            'task_template': self.goal_to_task.get(goal, ''),
            'goal': goal,
            'episode': self.episode,
            'step': self.step_index,
            'started_at': started_at,
            'step_s': round(step_s, 3),
            'llm_calls': len(durations),
            'action_llm_s': round(durations[0], 3) if len(durations) > 0 else '',
            'summary_llm_s': round(durations[1], 3) if len(durations) > 1 else '',
            'llm_total_s': round(llm_total, 3),
            'non_llm_s': round(step_s - llm_total, 3),
            'done': bool(getattr(result, 'done', False)),
            'action': action,
            'error': error,
        }
        new_file = not self.path.exists()
        with self.path.open('a', encoding='utf-8-sig' if new_file else 'utf-8', newline='') as target:
            writer = csv.DictWriter(target, fieldnames=self.FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f'[STEP TIME] episode {self.episode} step {self.step_index}: '
              f'{step_s:.1f}s total, {llm_total:.1f}s model, {step_s - llm_total:.1f}s other', flush=True)


def export_results(results, directory, model_calls, min_successes, model='gemma4e4b', context_overflows=0,
                   connection_retries=0):
    fields = ['task_template', 'goal', 'is_successful', 'episode_length', 'run_time', 'exception_info']
    rows = []
    for episode in results:
        row = {key: episode.get(key) for key in fields}
        error = row['exception_info']
        if isinstance(error, float) and math.isnan(error):
            row['exception_info'] = None
        rows.append(row)
    with (directory / 'results.csv').open('w', encoding='utf-8-sig', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    successes = sum(float(row['is_successful'] or 0) > .5 for row in rows)
    exceptions = sum(bool(row['exception_info']) for row in rows)
    summary = {
        'model': model,
        'agent': 'T3A (text UI tree)',
        'llm_location': 'physical Android phone, via local adb forward',
        'gui_location': 'Windows Pixel 6 / API 33 AndroidWorld emulator',
        'tasks_recorded': len(rows),
        'successes': successes,
        'task_success_rate': successes / len(rows) if rows else 0,
        'exceptions': exceptions,
        'model_calls_this_invocation': model_calls,
        'context_overflow_calls': context_overflows,
        'connection_retries': connection_retries,
        'acceptance_passed': bool(rows) and exceptions == 0 and successes >= min_successes,
    }
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def _disable_adb_server_restart():
    """Stop android_env from running `adb kill-server` after a failed adb command.

    WSL shares the single Windows adb server (ADB_SERVER_SOCKET). Killing it drops
    the `adb forward tcp:8080` rule that carries every request to the phone model,
    so a harmless action failure (e.g. open_app with an unknown name) would break
    the next model call. Retries still happen; only the server restart is skipped.
    """
    from absl import logging
    from android_env.components import adb_controller

    def _no_restart(self, timeout=None):
        logging.info('Skipping adb server restart (shared Windows adb server keeps the phone forward).')
        time.sleep(0.2)

    adb_controller.AdbController._restart_server = _no_restart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8080/v1')
    parser.add_argument('--model', default='gemma4e4b')
    parser.add_argument('--adb-path', default=shutil.which('adb'))
    parser.add_argument('--console-port', type=int, default=5554)
    parser.add_argument('--grpc-port', type=int, default=8554)
    parser.add_argument('--tasks', default='ClockStopWatchRunning', help='Comma-separated names, or all')
    parser.add_argument('--combinations', type=int, default=1)
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--output', type=Path, default=Path('runs/gemma4-smoke'))
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--min-successes', type=int, default=0)
    parser.add_argument('--phone-serial', default=None,
                        help='Phone adb serial; if set, a dropped model connection re-creates the adb forward and retries')
    args = parser.parse_args()
    if args.combinations < 1 or args.max_tokens < 1 or args.min_successes < 0:
        parser.error('combinations/max-tokens must be positive; min-successes must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output.glob('*.pkl.gz')):
        parser.error('Output already has task checkpoints. Choose a fresh output directory, or explicitly use --resume.')
    llm = PhoneGemma(args.base_url, args.model, args.output / 'llm_calls.jsonl', args.max_tokens, args.timeout)
    llm.phone_serial, llm.adb_path = args.phone_serial, args.adb_path
    llm.check_model()
    if args.probe_only:
        answer, _, _ = llm.predict('Reply with exactly: GEMMA_PHONE_OK')
        print(answer)
        print('PROBE COMPLETE: the local phone endpoint returned text.')
        return 0
    if not args.adb_path or not Path(args.adb_path).is_file():
        parser.error('adb was not found. Follow the WSL PATH step in the guide.')
    try:
        with socket.create_connection(('127.0.0.1', args.grpc_port), timeout=10):
            pass
    except OSError as error:
        raise RuntimeError('Cannot reach Windows emulator gRPC. Check WSL mirrored networking and emulator -grpc 8554.') from error

    # Delayed imports allow --probe-only to work without installing AndroidWorld.
    from absl import flags
    from android_world import checkpointer
    from android_world import registry
    from android_world import suite_utils
    from android_world.agents import t3a
    from android_world.env import env_launcher

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(['run_gemma_androidworld.py'])
    _disable_adb_server_restart()
    env = env_launcher.load_and_setup_env(
        console_port=args.console_port,
        grpc_port=args.grpc_port,
        adb_path=args.adb_path,
        emulator_setup=args.setup_only,
    )
    try:
        if args.setup_only:
            print('EMULATOR SETUP COMPLETE. Run the smoke task without --setup-only next.')
            return 0
        family = registry.TaskRegistry.ANDROID_WORLD_FAMILY
        task_map = registry.TaskRegistry().get_registry(family=family)
        tasks = None if args.tasks.lower() == 'all' else [name.strip() for name in args.tasks.split(',') if name.strip()]
        if tasks:
            unknown = set(tasks) - set(task_map)
            if unknown:
                raise ValueError(f'Unknown AndroidWorld task names: {sorted(unknown)}')
        suite = suite_utils.create_suite(
            task_map, n_task_combinations=args.combinations, seed=args.seed,
            tasks=tasks, use_identical_params=False, env=env,
        )
        suite.suite_family = family
        agent = t3a.T3A(env, llm, name='t3a_gemma4_phone')
        agent.transition_pause = None
        goal_to_task = {instance.goal: name for name, instances in suite.items() for instance in instances}
        StepTimer(args.output / 'step_times.csv', llm, goal_to_task).wrap(agent)
        results = suite_utils.run(
            suite, agent,
            checkpointer=checkpointer.IncrementalCheckpointer(str(args.output)),
            demo_mode=False,
        )
        summary = export_results(results, args.output, llm.calls, args.min_successes, model=llm.model,
                                 context_overflows=llm.context_overflows,
                                 connection_retries=llm.connection_retries)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f'Results: {args.output.resolve()}')
        return 0 if summary['acceptance_passed'] else 2
    finally:
        env.close()


if __name__ == '__main__':
    sys.exit(main())
