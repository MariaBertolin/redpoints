# REDPOINTS: TECHNICAL CHALLENGE
"Context: You are joining a brand protection platform that monitors online marketplaces for counterfeit product listings. One of the core problems we solve is detecting listings that are suspiciously similar to protected brand references — sellers often slightly alter product titles to avoid exact-match detection"

## STAGE 1
Classificar d'una llista de títols en:
    - No producte: ASSET_DISCARDED    
    - Producte: La resta

Proposta: Classificació binaria amb una regressió logística. Es podria utilitzar mBERT, però per limitacions de computació he optat per lo més lleuger possible. Així que utilitzaré TF-IDF+LogisticRegression (LR), més simple. 
- TF-IDF-> vectorització del text amb pesos segons freqüència
- LR -> classificació dels vectors en una de les dues classes

El threshold del LR es fixa prioritzant el recall, així que com millor Recall millor performance. 
Volem prioritzar els falsos positius perquè perdre un asset faria que no es pogués evaluar una potencial infracció en el stage2.

Seria interessant millorar-ho afegint correlacions semàntiques.

Paràmetres a ajustar: Threshold  i iteracions de la LR (podrien haver-hi més però ho he deixat lo més simple possible)


**=== Stage1 results ===**
| split | accuracy | precision | recall | f1 | loss |
|------|----------|-----------|--------|----|------|
| val  | 0.913102 | 0.913589  | 0.995444 | 0.952762 | 0.335431 |
| test | 0.913333 | 0.915503  | 0.993182 | 0.952762 | 0.346016 |

## SIMILARITY
Donada una consulta (un títol), buscar top-k títols semblants.
Més o menys mateixa estrategia que en el stage 1:
- TF-IDF-> per vectoritzar amb certs pesos 
- Cosine-similarity -> mesura similitud calculant el cosinus de l'angle entre dos vectors
- Ranking dels top-k resultats més similars

## STAGE 2
Donat un listing d'assets (passats per l'Stage 1), classificar els productes en:
- No infracció: INFRINGEMENT_DISCARDED
- Potencial infracció/Infracció: INFRINGEMENT_VALIDATED, INFRINGEMENT_CONFIRMED, ONFIRMATION_ON_HOLD, INFRINGEMENT_VERIFIED, CONFIRMATION_DISCARDED (Internament passarà a ser INFRINGEMENT_CONFIRMED)

Proposta: Classificació binaria combinant regressió logística amb un senyal de similarity. Igual que al Stage1 utilitzo TF-IDF + LogisticRegression per simplicitat i velocitat. A més, afegeixo una comprovació de similitud amb listings positius coneguts per reforçar la detecció quan un títol és molt semblant a altres.

- TF-IDF -> per vectoritzar amb certs pesos
- LR -> per estimar la probabilitat de que el listing sigui una infracció
- Similarity -> per mesurar similitud calculant el cosinus de l'angle entre dos vectors

Paràmetres a ajustar: Threshold de la Logistic Regression, Threshold de similarity, Iteracions de la LR

**=== Stage2 results ===**
| split | accuracy | precision | recall | f1 | loss |
|------|----------|-----------|--------|----|------|
| val  | 0.873171 | 0.911650 | 0.944130 | 0.927605 | 0.370861 |
| test | 0.879190 | 0.918513 | 0.943136 | 0.930661 | 0.365565 |

# Funcionament
## Creació entorn virual dins del projecte
`python3 -m venv .venv`

`source .venv/bin/activate`

`pip install -r requirements.txt`

## Execució avaluació
`python src/main.py --run_stage1 --run_similarity --run_stage2`

## Execució inference
`python src/inference.py --input_file data/listing.csv`

El fitxer d’entrada ha de ser un CSV on la primera columna contingui els títols dels listings (columna title).
El script carregarà els models entrenats i retornarà una classificació final per cada fila (ASSET_DISCARDED, INFRINGEMENT_CONFIRMED o INFRINGEMENT_DISCARDED).

# API Usage

**Per falta de coneixements en APIs i falta de temps, la implementació de la API no ha quedat finalitzada. És funcional, però no he pogut repassar-ho i validar el seu funcionament.**

The API provides endpoints to run the full asset-detection pipeline and query previous analyses.

- **POST `/analyze`**: Processes a listing title through Stage 1 (asset classification), similarity search, and Stage 2 (suspicion scoring). Returns whether it is an asset, the similarity score, and the top-k most similar reference listings.
- **GET `/listings/recent`**: Returns the last *N* analyzed listings.
- **GET `/listings/by-threshold`**: Retrieves listings whose score exceeds a given threshold in Stage 1 or Stage 2.
- **GET `/metadata`**: Returns model paths, datasets used for training/validation, and evaluation metrics.

All results are stored in a persistent SQLite database (`analysis.db`) to ensure they remain available across API restarts.

Start server:
`uvicorn api.main:app --reload`
`uvicorn api.main:app --reload`

Available at: 


http://127.0.0.1:8000/docs



