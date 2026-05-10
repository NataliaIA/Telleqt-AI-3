# Telleqt AI — классификация дефектов шва на пакетиках с кормом

## 1. Описание задачи

В рамках тестового задания необходимо было построить пайплайн, который определяет, является ли пример дефектным или хорошим.

Исходные данные:

- `train.zip` — обучающий датасет с размеченными примерами;
- `test.zip` — анонимные тестовые примеры без разметки;
- каждый пример — это папка с 4 изображениями одного и того же пакетика;
- дефект может быть виден:
  - только с передней стороны;
  - только с задней стороны;
  - только при одном типе освещения;
  - иногда сразу на нескольких изображениях.

В обучающем датасете примеры лежат в директориях:

```text
good/ — хороший пример, label = 0
bad/  — дефектный пример, label = 1
```

Требуемый результат — CSV-файл:

```text
sample_id,prediction
1,0
2,0
3,1
```

где:

sample_id — имя папки тестового примера;
prediction — 0 для good, 1 для bad.

## 2. Структура данных

После анализа структуры датасета:
```text
train:
  total samples: 755
  good: 401
  bad: 354

test:
  total samples: 241
```
Каждый пример содержит ровно 4 изображения.

Для train изображения имеют смысловой порядок:
```text
01 — front_barlight
02 — front_toplight
03 — back_barlight
04 — back_toplight
```
Для test структура выглядит так:
```text
test/
  1/
    01.jpg
    02.jpg
    03.jpg
    04.jpg
  2/
    01.jpg
    02.jpg
    03.jpg
    04.jpg
```
## 3. Основная идея решения

Задача решалась как multi-view binary classification.

Один физический объект представлен четырьмя изображениями:
```text
front + barlight
front + toplight
back + barlight
back + toplight
```
Так как дефект может быть виден только на одной стороне или только при одном типе света, модель обрабатывает все 4 изображения совместно.

Архитектура:
```text
4 изображения одного sample
        ↓
shared CNN encoder
        ↓
признаки каждого вида
        ↓
mean pooling + max pooling по видам
        ↓
binary classifier
        ↓
probability of defect
```
В качестве encoder используется EfficientNet-B0, предобученный на ImageNet.

Такой подход лучше, чем классифицировать каждую картинку отдельно, потому что итоговое решение принимается на уровне всего примера, а не отдельного изображения.

## 4. Почему используется Group Cross-Validation

Датасет собран в несколько разных сессий съёмки. Если делать обычный случайный train/validation split, в train и validation могут попасть очень похожие примеры из одной и той же сессии.

Чтобы оценка была честнее, используется:
```text
StratifiedGroupKFold
```
Группировка выполняется по source-директории съёмки.

Это позволяет проверять, насколько модель переносится на данные из другой сессии, а не просто запоминает особенности конкретного набора изображений.

## 5. Установка
```text
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
Для Windows:
```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
## 6. Подготовка данных

Ожидаемая структура проекта:
```text
project/
  data/
    train/
      BLTA_.../
        good/
          sample_x/
            01_....jpg
            02_....jpg
            03_....jpg
            04_....jpg
        bad/
          sample_y/
            01_....jpg
            02_....jpg
            03_....jpg
            04_....jpg

    test/
      1/
        01.jpg
        02.jpg
        03.jpg
        04.jpg
      2/
        01.jpg
        02.jpg
        03.jpg
        04.jpg
```
Данные не добавляются в репозиторий.

## 7. Обучение модели

Основной запуск:
```text
python -m telleqt_defects.train_cv \
  --train-root data/train \
  --out-dir runs/effnet_b0_group_cv \
  --cv group \
  --folds 4 \
  --epochs 20 \
  --batch-size 8 \
  --image-size 384 \
  --views all
```
После обучения сохраняются:
```text
runs/effnet_b0_group_cv/
  fold_0.pt
  fold_1.pt
  fold_2.pt
  fold_3.pt
  metrics.json
  threshold.txt
  threshold_report.csv
  oof_predictions.csv
  confusion_matrix.png
  pr_curve.png
  dataset_summary.json
  folds.csv
```
## 8. Генерация submission.csv

После обучения итоговый CSV для test создаётся командой:
```text
python -m telleqt_defects.predict \
  --test-root data/test \
  --model-dir runs/effnet_b0_group_cv \
  --out-csv submission.csv
```
Результат:
```text
sample_id,prediction
1,0
2,0
3,1
```
При inference используется ensemble по всем fold-моделям. Вероятности дефекта усредняются, после чего применяется выбранный threshold.

## 9. Метрики на кросс-валидации

Метрики рассчитаны на out-of-fold predictions.

Использовались:
```text
CV splitter: StratifiedGroupKFold
folds: 4
views: 01, 02, 03, 04
threshold strategy: best F1
selected threshold: 0.02684641
```
Итоговые метрики:
```text
Метрика	Значение
PR-AUC	0.9961
Recall по bad	0.9802
False Positive Rate	0.0399
Accuracy	0.9695
Precision по bad	0.9559
F1	0.9679
```
Confusion matrix:
```text
                 pred_good   pred_bad
true_good            385        16
true_bad               7       347
```
Интерпретация:

из 354 дефектных примеров модель нашла 347;
пропущено 7 дефектных примеров;
из 401 хорошего примера ошибочно забраковано 16;
recall по дефектам составил около 98%;
false positive rate составил около 4%.

Для задачи контроля качества такой operating point является полезным, потому что пропуск дефекта обычно дороже, чем лишняя отбраковка хорошего объекта.

## 10. PR-кривая и AUC

В рамках задания построена PR-кривая и рассчитан её AUC.

Полученное значение:
```text
PR-AUC = 0.9961
```
Высокое значение PR-AUC показывает, что модель хорошо ранжирует дефектные примеры выше хороших.

Файл с графиком:
```text
runs/effnet_b0_group_cv/pr_curve.png
```
## 11. Выбор threshold

Дополнительно был рассчитан отчёт по разным operating points.

Mode	Threshold	Recall bad	FPR	FN	FP
fixed_0.50	0.5000	0.9435	0.0150	20	6
target_recall_0.95	0.3468	0.9520	0.0200	17	8
best_f1	0.0268	0.9802	0.0399	7	16

В итоговом решении используется threshold, выбранный по out-of-fold предсказаниям.

Важно: фиксированный threshold 0.5 не всегда оптимален для нейросетевых моделей, особенно на небольших промышленных датасетах. Поэтому threshold подбирался на OOF-предсказаниях.


## Что было сделано по требованиям ТЗ
## 1. Построен пайплайн классификации good / bad

Реализован полный пайплайн:
```text
загрузка датасета
→ чтение 4 изображений одного sample
→ обучение multi-view модели
→ cross-validation
→ подбор threshold
→ inference на test
→ генерация submission.csv
```
## 2. Назначены метки 0 / 1 для test

Скрипт predict.py создаёт CSV-файл:
```text
sample_id,prediction
```
где:
```text
0 — good
1 — bad
```
## 3. Подготовлен код для GitHub

Код организован как Python-пакет:
```text
telleqt_defects/
  data.py
  model.py
  train_cv.py
  predict.py
  metrics.py
  error_analysis.py
  run_ablation.py
  gradcam.py
```
## 4. Подготовлен README

README описывает:
```text
структуру данных;
подход к решению;
архитектуру модели;
команды запуска;
формат результата;
метрики на кросс-валидации;
дополнительные эксперименты.
```
## 5. Посчитаны требуемые метрики

В рамках задания рассчитаны:
```text
confusion matrix;
recall;
false positive rate;
PR curve;
PR-AUC.
```
## Что было сделано сверх требований ТЗ
## 1. Multi-view подход вместо классификации отдельных изображений

Вместо обучения на отдельных картинках модель принимает решение на уровне всего примера из 4 изображений.

Это важно, потому что дефект может быть заметен только:
```text
на передней стороне;
на задней стороне;
при barlight;
при toplight;
сразу на нескольких изображениях.
```
## 2. Group-aware cross-validation

Для более честной оценки качества использован 
```text
StratifiedGroupKFold.
```
Модель валидируется не на случайно перемешанных изображениях, а с учётом групп съёмки. Это снижает риск переоценки качества из-за похожих условий внутри одной сессии.

## 3. Подбор production-oriented threshold

Кроме стандартного threshold 0.5, были рассчитаны разные operating points.

Это позволяет выбрать режим под задачу производства:
```text
меньше false positives;
или выше recall по дефектам.
```
Для контроля качества особенно важен высокий recall, так как пропуск дефекта обычно дороже лишней отбраковки хорошего пакета.

## 4. Fold ensemble на inference

Для test-предсказаний используется ансамбль моделей, обученных на разных fold-ах.

Это делает итоговые предсказания стабильнее, чем использование одной модели.

## 5. Error analysis

Добавлен отдельный скрипт для анализа ошибок:
```text
python -m telleqt_defects.error_analysis \
  --train-root data/train \
  --run-dir runs/effnet_b0_group_cv \
  --views all \
  --max-per-type 50
```
Он сохраняет:
```text
false positives
false negatives
error_report.csv
error_summary.csv
```
Это позволяет вручную посмотреть, какие дефекты модель пропускает и какие хорошие примеры похожи на bad.

## 6. Ablation study по видам изображений

Добавлен скрипт для сравнения качества на разных наборах изображений:
```text
только 01
только 02
только 03
только 04
только front
только back
только barlight
только toplight
все 4 вида
```
Запуск:
```text
python -m telleqt_defects.run_ablation \
  --train-root data/train \
  --out-dir runs/ablation \
  --cv group \
  --folds 4 \
  --epochs 10 \
  --batch-size 8 \
  --image-size 384
```
Это помогает понять, какие стороны пакета и какие условия освещения наиболее информативны для поиска дефектов.

## 7. Grad-CAM / explainability

Добавлена возможность построить Grad-CAM heatmap для визуальной проверки того, куда смотрит модель.

Пример запуска:
```text
python -m telleqt_defects.gradcam \
  --train-root data/train \
  --model-dir runs/effnet_b0_group_cv \
  --from-oof confident_bad \
  --top-k 8
```
Это полезно для sanity check: модель должна смотреть на область шва или визуальные признаки дефекта, а не на случайные элементы фона.

## Основной вывод

Решение показывает высокое качество на out-of-fold валидации:
```text
PR-AUC: 0.9961
Recall bad: 0.9802
False Positive Rate: 0.0399
```
Модель пропустила только 7 дефектных примеров из 354 и ошибочно забраковала 16 хороших примеров из 401.

Итоговый пайплайн ориентирован не только на получение метрик, но и на промышленное использование:
```text
учитываются все 4 изображения объекта;
используется group-aware validation;
threshold подбирается на OOF-предсказаниях;
есть режим высокого recall;
сохраняются ошибки для визуального анализа;
добавлены ablation study и Grad-CAM для интерпретации результата.
```
