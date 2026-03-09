# Evaluation

## Stage 1 — Product classification
Aquest stage funciona com un filtre inicial del pipeline, eliminant listings que no són productes abans de passar a les etapes següents.

### Resultats

| split | accuracy | precision | recall | f1 | loss |
|------|---------|----------|-------|------|------|
| val  | 0.913102 | 0.913589 | 0.995444 | 0.952762 | 0.335431 |
| test | 0.913333 | 0.915503 | 0.993182 | 0.952762 | 0.346016 |

### Consideracions sobre les mètriques

En aquest stage es prioritza maximitzar el recall.  
Un fals negatiu implicaria descartar un asset que podria ser una infracció i que ja no arribaria al Stage 2.

Per aquest motiu s’ha utilitzat un threshold relativament baix (0.2), que permet mantenir un recall molt alt (~0.99) i acceptar més falsos positius, que seran analitzats posteriorment.

Per trobar aquest threshold s'ha fet una funció (evaluate_threshold), que executava el resultat del training i la validació per un rang de diferents llindars. Concretament de 0.10 a 0.90, fent steps de 0.05. I finalment vaig decidir usar el 0.2 ja que era el valor que donava una millor recall, sense perdre de les altres mètriques (sense que les altres estiguessin per sota de 0.8).


### Sobre les dades

El dataset presenta diverses característiques que influeixen al model implementat: el desbalanceig de classes i que sigui multilingüe.
La majoria de listings corresponen a productes, mentre que ASSET_DISCARDED és menys freqüent. Per compensar-ho s’ha utilitzat: el `class_weight=balanced`
Els títols apareixen en diferents idiomes i amb variacions ortogràfiques. Per aquest motiu s’ha optat per treballar amb n-grames de rang 3-5, amb normalitzacions d'accents (a part de posar tot en mínuscules).

## Stage 2 — Suspicion scoring

Aquest stage analitza els listings que han passat el Stage 1 i calcula el nivell de similitud amb listings de referència, amb l’objectiu d’identificar possibles infraccions.

### Resultats
| split | accuracy | precision | recall | f1 | loss |
|------|----------|-----------|--------|----|------|
| val  | 0.873171 | 0.911650 | 0.944130 | 0.927605 | 0.370861 |
| test | 0.879190 | 0.918513 | 0.943136 | 0.930661 | 0.365565 |

### Consideracions sobre les mètriques

S'ha utilitzat `evaluate_threshold` per provar diferents llindars de decisió.  

Avaluar diferents llindars permet veure el compromís entre **precision i recall** i escollir un valor que mantingui un bon equilibri entre detectar possibles infraccions i evitar massa falsos positius. De totes maneres segueixo prioritzant el recall.

### Sobre les dades

Les dades d’aquest stage venen heredades del Stage 1, ja que només s’analitzen els listings classificats com a producte!!

Per tant, alguns errors del Stage 1 es poden propagar al Stage 2, tot i que el Stage 1 està configurat per prioritzar el recall i minimitzar la pèrdua de possibles casos d’infracció.
