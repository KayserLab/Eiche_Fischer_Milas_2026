"""Conservative dependency scheduling for independently audited pipeline stages."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import ctypes
from datetime import datetime
import os
import threading
import time


def stage_resources(step, plate_workers, analysis_workers):
    # Everything else is exclusive: plotters can rebuild shared caches even
    # when the explicit producer dependencies have already completed.
    if step.name in {'colony_masks', 'clone_tracking', 'clean_sensitive'}:
        return 'microscopy', plate_workers
    if step.name == 'escape_inputs':
        return 'microscopy', analysis_workers
    if step.name in {'simulation_overview', 'simulation_points'}:
        return 'simulation', analysis_workers
    if step.name in {'cell_size_density', 'fig3_cell_size'}:
        return 'cell_size', 1
    if step.name in {'fig3_master_120', 'fig3_master_160'}:
        return step.name, 1
    return None, 1


class CapacityMonitor:
    """Sample system-wide CPU use and available RAM without extra packages."""
    def __init__(self):
        self.previous = None

    def sample(self):
        if os.name != 'nt':
            return None, None  # Unknown capacity: run serially.
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in (
                    'total_phys', 'avail_phys', 'total_page', 'avail_page',
                    'total_virtual', 'avail_virtual', 'avail_extended')]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        api = ctypes.windll.kernel32
        if not api.GlobalMemoryStatusEx(ctypes.byref(memory)) or not api.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None, None
        ticks = lambda value: (value.dwHighDateTime << 32) | value.dwLowDateTime
        current = ticks(idle), ticks(kernel) + ticks(user)
        cpu = None
        if self.previous is not None:
            idle_delta, total_delta = (now - old for now, old in zip(current, self.previous))
            if total_delta > 0:
                cpu = max(0., min(100., 100 * (1 - idle_delta / total_delta)))
        self.previous = current
        return cpu, memory.avail_phys / 1024 ** 3


def execute_plan(commands, env, log_dir, report, save_report, run_step, announce,
                 *, max_stages, worker_budget, plate_workers, analysis_workers,
                 min_free_gb=32, max_cpu=85, monitor=None):
    """Launch only ready stages with disjoint resources; cancel on failure."""
    monitor = monitor or CapacityMonitor()
    selected = {step.name for step, _ in commands}
    complete, started = set(), set()
    active = {}
    stop = threading.Event()
    interrupted = False
    failed = False

    def finish(future, step, entry, began):
        nonlocal failed
        try:
            code = future.result()
            entry.update(exit_code=code, status='complete' if code == 0 else 'failed')
        except Exception as error:
            entry.update(status='failed', error=str(error))
        entry['seconds'] = time.monotonic() - began
        if entry['status'] == 'complete':
            complete.add(step.name)
        else:
            failed = True
            stop.set()
            announce(f"FAILED: {step.name} (exit {entry.get('exit_code', 'error')})")
            if entry.get('error'):
                announce(entry['error'])
        save_report()

    with ThreadPoolExecutor(max_workers=max_stages) as pool:
        try:
            while len(complete) < len(commands) and not failed:
                cpu, free_gb = monitor.sample()
                for index, (step, command) in enumerate(commands, 1):
                    if step.name in started or (set(step.requires) & selected) - complete:
                        continue
                    if len(active) >= max_stages:
                        break
                    group, cost = stage_resources(step, plate_workers, analysis_workers)
                    if active:
                        groups = {item[3] for item in active.values()}
                        used = sum(item[4] for item in active.values())
                        if (group is None or None in groups or group in groups
                                or used + cost > worker_budget):
                            continue
                        if (cpu is None or free_gb is None or cpu >= max_cpu
                                or free_gb < min_free_gb + 2 * cost):
                            continue
                    entry = {'step': step.name, 'command': command, 'status': 'running',
                             'started_at': datetime.now().isoformat(timespec='seconds'),
                             'log_path': str(log_dir / f'{index:02d}_{step.name}.log')}
                    report['steps'].append(entry)
                    started.add(step.name)
                    save_report()
                    announce(f'\n[{index}/{len(commands)}] Running {step.name}')
                    child_env = env.copy()
                    child_env['EICHE_FISCHER_CAPTURE_PLOTS'] = '1' if step.group.startswith('fig') or step.group == 'suppl' or step.name == 'raw_demo_report' else '0'
                    child_env['MPLCONFIGDIR'] = str(log_dir / f'matplotlib_{step.name}')
                    if max_stages > 1:
                        child_env['PIPELINE_STAGE_LABEL'] = step.name
                        # Internal pools already account for the requested workers.
                        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                    'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
                            child_env[key] = '1'
                    began = time.monotonic()
                    future = pool.submit(run_step, command, child_env,
                                         log_dir / f'{index:02d}_{step.name}.log', stop_event=stop)
                    active[future] = step, entry, began, group, cost
                if not active:
                    if len(complete) == len(commands):
                        break
                    raise ValueError('No runnable stages; check dependencies')
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    step, entry, began, _, _ = active.pop(future)
                    finish(future, step, entry, began)
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
        except BaseException:
            stop.set()
            raise
        finally:
            # run_step kills its own process tree on cancellation, including pools.
            for future, (step, entry, began, _, _) in active.items():
                try:
                    code = future.result()
                    entry.update(exit_code=code, status='complete' if code == 0 else 'interrupted')
                    if code == 0:
                        complete.add(step.name)
                except Exception as error:
                    entry.update(status='interrupted', error=str(error))
                entry['seconds'] = time.monotonic() - began
            save_report()
    if interrupted or failed:
        report['status'] = 'interrupted' if interrupted else 'failed'
        # Completion order can differ from plan order. Resume at the earliest
        # unfinished stage so an interrupted sibling isn't accidentally omitted.
        resume = next(step.name for step, _ in commands if step.name not in complete)
        report['resume_from'] = resume
        save_report()
        announce(f'Stopped. Resume with --from-step {resume}. Logs: {log_dir}')
        return 130 if interrupted else 1
    report['status'] = 'complete'
    save_report()
    announce(f'Completed all {len(commands)} selected steps. Logs: {log_dir}')
    return 0
