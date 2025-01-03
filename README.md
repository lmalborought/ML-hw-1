# ML-hw-1
## Что было сделано:
1. Был проведен EDA
2. Была проведена обработка данных
3. Были обучены модели

## Результаты:
1. Классическая линейная регрессия
   * MSE для train: 117125315893.244
   * MSE для test: 245703643929.17392
   * R2 для train: 0.5913829372433583
   * R2 для tes: 0.5725618681968949
2. Классическая линейная регрессия со стандартизацией
   * MSE для train: 117125315893.24414
   * MSE для test: 245703643929.1572
   * R2 для train: 0.5913829372433579
   * R2 для tes: 0.572561868196924
3. Lasso регрессия
    * MSE для train: 117125315895.49362
   * MSE для test: 245704013786.58936
   * R2 для train: 0.5913829372355102
   * R2 для tes: 0.5725612247747618
4. Переборпо сетке (c 10-ю фолдами) оптимальные параметры для Lasso-регрессии
   * alpha = 10
   * MSE для train: 117125316118.00153
   * MSE для test: 245707201344.19876
   * R2 для train: 0.5913829364592431
   * R2 для tes: 0.5725556795429216
5. Перебор по сетке (c 10-ю фолдами) оптимальные параметры для ElasticNet регрессии
    * alpha=10, l1_ratio=1.0
   * MSE для train: 117125316118.00153
   * MSE для test: 245707201344.19876
   * R2 для train: 0.5913829364592431
   * R2 для tes: 0.5725556795429216

## Что не было сделано:
- Не был предобработан столбец torque
- Не был предобработан столбец name, что могло дать сильный прирост по метрикам качества
