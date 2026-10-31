import math
import random
import statistics
import time


import plotly.graph_objects as go
import streamlit as st


Number = int | float


def stupid_sort(a: list[Number]) -> tuple[list[Number], int, int]:
    comparisons = swaps = 0
    i = 0
    while i < len(a) - 1:
        comparisons += 1
        if a[i] > a[i + 1]:
            a[i], a[i + 1] = a[i + 1], a[i]
            swaps += 1
            i = 0
        else:
            i += 1
    return a, comparisons, swaps


def bubble_sort(a: list[Number]) -> tuple[list[Number], int, int]:
    comparisons = swaps = 0
    for end in range(len(a) - 1, 0, -1):
        changed = False
        for i in range(end):
            comparisons += 1
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                swaps += 1
                changed = True
        if not changed:
            break
    return a, comparisons, swaps


def quick_sort(a: list[Number]) -> tuple[list[Number], int, int]:
    comparisons = swaps = 0
    stack = [(0, len(a) - 1)]
    while stack:
        left, right = stack.pop()
        if left >= right:
            continue
        middle = (left + right) // 2
        if middle != right:
            a[middle], a[right] = a[right], a[middle]
            swaps += 1
        pivot = a[right]
        boundary = left
        for i in range(left, right):
            comparisons += 1
            if a[i] <= pivot:
                if i != boundary:
                    a[i], a[boundary] = a[boundary], a[i]
                    swaps += 1
                boundary += 1
        if boundary != right:
            a[boundary], a[right] = a[right], a[boundary]
            swaps += 1
        if boundary + 1 < right:
            stack.append((boundary + 1, right))
        if left < boundary - 1:
            stack.append((left, boundary - 1))
    return a, comparisons, swaps


def prepare_gpu(values: list[Number], cp):
    integers = all(type(x) is int for x in values)
    if integers and any(x < -(2**63) or x >= 2**63 for x in values):
        raise ValueError("Для GPU целые числа должны помещаться в int64")
    if not integers and any(type(x) is int and abs(x) > 2**53 for x in values):
        raise ValueError("Смешанный массив должен помещаться в точность float64")
    n = len(values)
    m = 1
    while m < n:
        m *= 2
    dtype = cp.int64 if integers else cp.float64
    a = cp.asarray(values, dtype=dtype)
    if n < m:
        fill = 2**63 - 1 if integers else cp.inf
        a = cp.concatenate((a, cp.full(m - n, fill, dtype=dtype)))
    return a


def bitonic_gpu(a, cp):
    n = len(a)
    ids = cp.arange(n)
    comparisons = 0
    swaps = cp.zeros((), dtype=cp.int64)

    size = 2

    while size <= n:
        step = size // 2

        while step:
            other = a[ids ^ step]

            take_min = ((ids & size) == 0) == ((ids & step) == 0)

            new_a = cp.where(
                take_min,
                cp.minimum(a, other),
                cp.maximum(a, other)
            )

            swaps += cp.count_nonzero(new_a != a) // 2
            comparisons += n // 2

            a = new_a
            step //= 2

        size *= 2

    return a, comparisons, int(swaps.get())


SORTS = {"Глупая": stupid_sort, "Пузырёк": bubble_sort, "QuickSort": quick_sort}
NAMES = list(SORTS) + ["Bitonic GPU"]
DISTRIBUTIONS = ["Случайный", "Отсортированный", "Обратный", "Почти отсортированный", "Дубликаты"]


def run_sort(name: str, values: list[Number]):
    if name != "Bitonic GPU":
        a = values.copy()
        start = time.perf_counter()
        result, comparisons, swaps = SORTS[name](a)
        elapsed = (time.perf_counter() - start) * 1000
        return result, comparisons, swaps, elapsed

    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CUDA-видеокарта не обнаружена")
    except ImportError as error:
        raise RuntimeError("Для GPU нужен CuPy и видеокарта NVIDIA") from error
    a = prepare_gpu(values, cp)
    cp.cuda.get_current_stream().synchronize()
    start = time.perf_counter()
    result, comparisons, swaps = bitonic_gpu(a, cp)
    cp.cuda.get_current_stream().synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    return cp.asnumpy(result[:len(values)]).tolist(), comparisons, swaps, elapsed


def generate(n: int, kind: str, seed: int) -> list[int]:
    rng = random.Random(seed)
    if kind == "Отсортированный":
        return list(range(n))
    if kind == "Обратный":
        return list(range(n, 0, -1))
    if kind == "Дубликаты":
        return [rng.randint(1, 5) for _ in range(n)]
    if kind == "Почти отсортированный":
        a = list(range(n))
        indices = rng.sample(range(n), max(2, n // 20))
        values = [a[i] for i in indices]
        rng.shuffle(values)
        for i, value in zip(indices, values):
            a[i] = value
        return a
    return [rng.randrange(n * 2) for _ in range(n)]


def fit_shape(name: str, n: int, kind: str) -> float:
    if name in ("Глупая", "Пузырёк") and kind == "Отсортированный":
        return float(n)
    if name == "Глупая":
        return float(n**3)
    if name == "Пузырёк":
        return float(n**2)
    if name == "QuickSort":
        return n * math.log2(n)
    m = 1 << (n - 1).bit_length()
    return m * math.log2(m) ** 2


def results_chart(rows: list[dict], distribution: str, scale: str, fitted: bool):
    fig = go.Figure()
    names = list(dict.fromkeys(row["Алгоритм"] for row in rows))
    for name in names:
        group = [r for r in rows if r["Алгоритм"] == name and r["Данные"] == distribution]
        sizes = list(dict.fromkeys(r["N"] for r in group))
        x, y, errors = [], [], []
        for n in sizes:
            times = [r["Время, мс"] for r in group if r["N"] == n]
            x.append(n)
            y.append(statistics.mean(times))
            errors.append(statistics.stdev(times) if len(times) > 1 else 0)
        if not x:
            continue
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name,
                                 error_y=dict(type="data", array=errors, visible=True)))
        if fitted:
            shape = [fit_shape(name, n, distribution) for n in x]
            c = sum(t * f for t, f in zip(y, shape)) / sum(f * f for f in shape)
            fig.add_trace(go.Scatter(x=x, y=[c * f for f in shape], mode="lines",
                                     name=name + " (теория)", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Размер массива N", yaxis_title="Время, мс",
                      legend_title="Алгоритм")
    fig.update_xaxes(type="log" if scale == "Log-Log" else "linear", showgrid=True)
    fig.update_yaxes(type="log" if scale == "Log-Log" else "linear", showgrid=True)
    return fig


def bubble_frames(values: list[int]):
    a = values.copy()
    frames = [(a.copy(), -1)]
    for end in range(len(a) - 1, 0, -1):
        changed = False
        for i in range(end):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                changed = True
            frames.append((a.copy(), i))
        if not changed:
            break
    return frames




st.set_page_config(page_title="Sorting Lab", layout="wide")
st.sidebar.page_link("app.py", label="Home")
st.sidebar.page_link("pages/sorts.py", label="Работа_Максима_Копылова")
st.title("Sorting Lab")

tab1, tab2, tab3 = st.tabs(["Проверка", "Эксперимент", "Шаги пузырька"])

with tab1:
    name = st.selectbox("Алгоритм", NAMES)
    text = st.text_input("Числа через запятую", "8, 3, 7, 1, 6, 2, 5, 4")
    if st.button("Отсортировать", type="primary"):
        try:
            data = [float(x) if any(c in x.lower() for c in (".", "e")) else int(x)
                    for x in text.split(",") if x.strip()]
            if not all(math.isfinite(x) for x in data):
                raise ValueError("Нужны конечные числа")
            if name == "Глупая" and len(data) > 500:
                raise ValueError("Для глупой сортировки максимум 500 элементов")
            result, comparisons, swaps, milliseconds = run_sort(name, data)
            st.write("**Результат:**", result)
            a, b, c = st.columns(3)
            a.metric("Время", f"{milliseconds:.3f} мс")
            b.metric("Сравнения", comparisons)
            c.metric("Перестановки", swaps)
        except Exception as error:
            st.error(str(error))

with tab2:
    selected = st.multiselect("Алгоритмы", NAMES, default=NAMES[:3])
    distribution = st.selectbox("Входные данные", DISTRIBUTIONS + ["Все пять типов"])
    max_n = st.number_input("Максимальный размер N", min_value=20, max_value=1000000, value=100, step=20)
    repeats = st.slider("Повторений", 5, 10, 5)
    if st.button("Измерить"):
        rows = []
        kinds = DISTRIBUTIONS if distribution == "Все пять типов" else [distribution]
        sizes = list(dict.fromkeys(max(2, int(max_n) * k // 5) for k in range(1, 6)))
        try:
            if "Bitonic GPU" in selected:
                run_sort("Bitonic GPU", [4, 1, 3, 2])
            with st.spinner("Измеряем..."):
                for kind in kinds:
                    for n in sizes:
                        for repeat in range(repeats):
                            data = generate(n, kind, repeat + 100)
                            for method in selected:
                                if method == "Глупая" and n > 500:
                                    continue
                                if method == "Пузырёк" and n > 10000:
                                    continue
                                if method == "QuickSort" and kind == "Дубликаты" and n > 10000:
                                    continue
                                output, comparisons, swaps, elapsed = run_sort(method, data)
                                if any(output[i] > output[i + 1] for i in range(len(output) - 1)):
                                    raise RuntimeError("Некорректная сортировка: " + method)
                                rows.append({"Алгоритм": method, "Данные": kind, "N": n,
                                             "Повтор": repeat + 1, "Время, мс": elapsed,
                                             "Сравнения": comparisons, "Перестановки": swaps})
            st.session_state["measurements"] = rows
        except Exception as error:
            st.error(str(error))
    if st.session_state.get("measurements"):
        rows = st.session_state["measurements"]
        kind = st.selectbox("Данные на графике", list(dict.fromkeys(r["Данные"] for r in rows)))
        scale = st.radio("Масштаб", ["Линейный", "Log-Log"], horizontal=True)
        fitted = st.checkbox("Теоретическая аппроксимация", value=True)
        st.plotly_chart(results_chart(rows, kind, scale, fitted), use_container_width=True)
        st.dataframe(rows, hide_index=True, use_container_width=True)
        import csv
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Скачать CSV", output.getvalue().encode("utf-8-sig"),
                           "measurements.csv", "text/csv")

with tab3:
    n = st.slider("Число элементов", 20, 30, 20)
    frames = bubble_frames(generate(n, "Случайный", 42))
    step = st.slider("Шаг сравнения", 0, len(frames) - 1)
    values, active = frames[step]
    colors = ["orange" if i in (active, active + 1) and active >= 0 else "steelblue"
              for i in range(n)]
    fig = go.Figure(go.Bar(x=list(range(n)), y=values, marker_color=colors))
    fig.update_layout(xaxis_title="Индекс", yaxis_title="Значение")
    st.plotly_chart(fig, use_container_width=True)
