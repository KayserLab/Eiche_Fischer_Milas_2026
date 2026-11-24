"""Independent-plate scheduling for the figure pipeline."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed


def run_plates(function, plates, workers: int, *, label: str, **kwargs):
    """Only independent plates run concurrently; exceptions propagate to the runner."""
    plates = list(plates)
    if workers < 1:
        raise ValueError("Plate workers must be positive")
    count = min(workers, len(plates))
    print(f"[{label}] using {count} plate worker(s)", flush=True)
    if count <= 1:
        for plate in plates:
            function(plate, **kwargs)
    else:
        with ProcessPoolExecutor(max_workers=count) as pool:
            futures = {pool.submit(function, plate, **kwargs): plate for plate in plates}
            for future in as_completed(futures):
                future.result()
