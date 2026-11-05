"""Run AndroidWorld T3A with Gemma served by the USB-connected Android phone.

The AndroidWorld task environment is the official API-33 emulator on Windows.
Only the model executes on the physical phone. No cloud-model fallback exists.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import socket
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
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

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
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'Phone llama-server HTTP {error.code}: {detail}') from error
        text = result['choices'][0]['message'].get('content')
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError('Empty model answer. Check start_phone.sh uses --reasoning-budget 0 and inspect the phone log.')
        self.calls += 1
        record = {
            'call': self.calls,
            'endpoint': request.full_url,
            'elapsed_s': time.perf_counter() - started,
            'prompt': text_prompt,
            'response': result,
        }
        with self.log_path.open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
        return text, None, result


def export_results(results, directory, model_calls, min_successes, model='gemma4e4b'):
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
        'acceptance_passed': bool(rows) and exceptions == 0 and successes >= min_successes,
    }
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


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
    args = parser.parse_args()
    if args.combinations < 1 or args.max_tokens < 1 or args.min_successes < 0:
        parser.error('combinations/max-tokens must be positive; min-successes must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output.glob('*.pkl.gz')):
        parser.error('Output already has task checkpoints. Choose a fresh output directory, or explicitly use --resume.')
    llm = PhoneGemma(args.base_url, args.model, args.output / 'llm_calls.jsonl', args.max_tokens, args.timeout)
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
        results = suite_utils.run(
            suite, agent,
            checkpointer=checkpointer.IncrementalCheckpointer(str(args.output)),
            demo_mode=False,
        )
        summary = export_results(results, args.output, llm.calls, args.min_successes, model=llm.model)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f'Results: {args.output.resolve()}')
        return 0 if summary['acceptance_passed'] else 2
    finally:
        env.close()


if __name__ == '__main__':
    sys.exit(main())
