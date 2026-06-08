# neural-ode-resnet-2
Course project: Discrete–continuous correspondence in Residual Networks, Neural ODE and adjoint methods

# ResNet, Neural ODE и обратимость: дискретная динамика и вычисление градиентов

Данный репозиторий содержит код для курсовой работы, посвящённой анализу связи между:
- остаточными нейронными сетями (ResNet),
- численными схемами решения ОДУ,
- моделями Neural Ordinary Differential Equations (Neural ODE),
- обратимостью дискретных динамических систем,
- и методами вычисления градиентов (backpropagation и adjoint method).

Основная цель работы — исследовать, при каких условиях глубокие residual-архитектуры позволяют:
- устойчиво восстанавливать скрытые состояния,
- и вычислять градиенты без хранения всей траектории,
- без потери точности из-за численной нестабильности.

---

# Структура репозитория

```text
.
├── experiment1_reconstruction.py   # восстановление скрытых состояний (обратимость)
├── experiment2_methods.py          # сравнение Euler / RK2 / Neural ODE
├── experiment3_gradients.py        # backprop vs adjoint vs discrete adjoint
├── ablation.py                     # устойчивость: h, depth, Lipschitz
├── models.py                       # ResNet (Euler), RK2, ODE-динамика
├── utils.py                        # метрики, логирование, графики
├── results/                        # сохранённые ошибки и графики
└── README.md

```
---

## Описание экспериментов
### 1. Эксперимент: динамика и обратимость (experiment1_trajectories.py)
Исследуется линейная динамическая система и её численные аппроксимации:

$x_{k+1} = x_k + h f(x_k)$

и её обратимость при восстановлении состояния назад по глубине.

Сравниваются:
- Euler (ResNet)
- Heun / RK2

Анализируются:
- ошибка прямой траектории
- ошибка восстановления (reconstruction error)

---

### 2. Эксперимент: классификация 2D-спиралей (experiment2_spirals.py)
Решается задача бинарной классификации на синтетическом датасете спиралей.

Сравниваются модели:

- ResNet (Euler blocks)
- ResNet (Heun / RK2 blocks)
- Neural ODE (fixed-step solver)
- Neural ODE (adaptive solver)

Метрики:
- accuracy
- time of training
- memory usage (approx)

---

## 3. Эксперимент: анализ градиентов (experiment3_gradients.py)
Исследуется разница между:

- стандартным backpropagation
- memory-free восстановлением скрытых состояний

Анализируется:
- ошибка градиента
- влияние ошибки реконструкции на обучение

---

## Абляционный анализ (ablation.py)

Исследуются режимы устойчивости:

- шаг h
- глубина сети (число блоков)
- липшицевость векторного поля
- тип численной схемы (Euler / RK2 / ODE solver)

Проверяется:
- устойчивость обратного прохода
- накопление ошибки при восстановлении
- условия применимости memory-free обучения

---

## Связь с курсовой работой

Эксперименты соответствуют теоретической части работы:

- ResNet как явная схема Эйлера
- Neural ODE как непрерывный предел
- обратимость residual-шагов
- сопряжённые методы и вычисление градиентов
- устойчивость memory-free восстановления

Основной исследуемый вопрос:

> когда глубинные модели позволяют экономить память без потери устойчивости и точности градиента

---

## Установка зависимостей
Рекомендуется использовать виртуальное окружение.

```bash
pip install -r requirements.txt
```

## Запуск экспериментов
```bash
python experiment1_trajectories.py
python experiment2_spirals.py
python experiment3_gradients.py
python ablation.py
```

Результаты сохраняются в папке в корневой папке
Визуальные выполненные результаты программ уже хранятся в папке results/

---

## Требования к среде (Requirements / Environment)

Для воспроизведения экспериментов достаточно стандартной Python-среды.

Рекомендуемая конфигурация:

- **Python:** 3.10+
- **PyTorch:** 2.x
- **Устройства:** CPU или GPU (CUDA поддерживается, но не обязателен)

### Основные зависимости:

- torch
- torchdiffeq
- numpy
- matplotlib
- pandas

---

## Литература

[1] Chen et al. Neural Ordinary Differential Equations  
[2] Haber, Ruthotto. Stable Architectures for Deep Neural Networks  
[3] Sander et al. Do Residual Neural Networks discretize Neural ODEs?  
[4] Chang et al. Reversible Architectures for Arbitrarily Deep Residual Networks  
[5] torchdiffeq library  
