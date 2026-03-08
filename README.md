# REDPOINTS CHALLENGE

# STAGE 1
Classificar d'una llista de títols:
    - No producte(asset): ASSET_DISCARDED    
    - Producte: La resta

Proposta: Classificació binaria amb una regressió logística. Es podria utilitzar mBERT, però el meu processador va curt i no hi he treballat. Així que utilitzaré TF-IDF+LogisticRegression, més simple (no necessito dataloader, forward, loss.backward...). 
- TF-IDF-> per vectoritzar amb certs pesos 
- LR -> per delimitar o no aquests vectors en una classe u altra, amb un threshold que prioritzi la recall.

Volem prioritzar els falsos positius, així que com millor Recall millor performance. Perdre un asset faria que no es pogués evaluar una potencial infracció en el stage2.

Paràmetres a ajustar: Threshold  i iteracions de la LR (podrien haver-hi més però ho he deixat lo més simple possible)

# SIMILARITY
Donada una consulta dún títol, buscar top-k títols semblants
Més o menys mateixa estrategia que en el stage 1:
- TF-IDF-> per vectoritzar amb certs pesos 
- Cosine-similarity -> mesura similitud calculant el cosinus de l'angle entre dos vectors
- Ranking dels top-k

# STAGE 2
Classificar d'una llista de productes detectats al stage 1 si poden correspondre a una possible infracció o no.

Proposta: Model de classificació sobre els títols que ja han passat el filtre del stage 1. L’objectiu és estimar si un producte és sospitós d’infracció basant-se en patrons textuals del títol.

# Arquitectura
redpoints/
│
├── data/
│   └── reference_listing.csv
│   └── labels.csv
│
├── stages/
│   ├── stage1_model.pkl
│   └── stage1_vectorizer.pkl
│
├── src/
│   ├── dataset.py      
│   ├── stage1.py        
│   ├── train.py        
│   └── predict.py      
│
├── requirements.txt
└── README.md

## Creació entorn virual dins del projecte
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt