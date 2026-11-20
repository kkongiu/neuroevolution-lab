# M1-P — Rapporto empirico sulla plasticità nell'apprendimento continuo

**Data:** 20 novembre 2026  
**Esecuzione:** [GitHub Actions #44818836611](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44818836611)  
**Codice sperimentale:** commit `85efc1e756307fa2ba44be17e643bc037e897bba`  
**Protocollo congelato:** [PLASTICITY_PROTOCOL.md](../docs/PLASTICITY_PROTOCOL.md)  
**Stato:** 24 seed completati con successo, dataset e grafico disponibili nell'artefatto `plasticity-m1p-24-seeds` (ID 14670018975; scadenza prevista 20 dicembre 2026).

## Sintesi

Il sistema che conserva integralmente i pesi attraverso compiti successivi ottiene **AUC media 0,7052**, contro **0,7510** della rete nuova a ogni compito: differenza appaiata **−4,59 punti percentuali**.

La sostituzione periodica del 20% delle unità nascoste meno attive porta l'AUC a **0,7287**, recuperando **2,35 pp** rispetto al modello continuo; il rinnovo casuale delle stesse unità raggiunge **0,7199** e recupera **1,47 pp**.

**Interpretazione:** evidenza di *trasferimento negativo / minore apprendibilità nel protocollo sequenziale di questo esperimento*. Non è dimostrato un declino cumulativo monotono della plasticità, né una proprietà generale di tutte le reti neurali. Gli agenti non scelgono ancora come auto-modificarsi.

## Disegno

- 24 sequenze di seed indipendenti, dal 100 al 123.
- Per ogni seed: **8 compiti** successivi (linea, XOR, cerchio, strisce, ripetuti con parametri nuovi).
- Per ogni compito: **128 aggiornamenti SGD**, batch da 2 esempi, stessi dati di addestramento per tutte le condizioni.
- MLP ReLU con 24 unità nascoste; accuratezza misurata su 96 esempi distinti, ai passi 0, 32, 64, 96 e 128.
- Quattro bracci: **continual** (tutti i pesi mantenuti), **fresh** (nuova inizializzazione), **low_activity_reset** (rinnovo del 20% delle unità meno attive sui *precedenti dati di training*) e **random_reset** (stesso numero di unità casuali).
- Primo compito: le quattro reti iniziano **identiche**. I dati sono aggregati sui compiti **1–7**, escludendo il primo.
- La metrica primaria è l'**area normalizzata sotto la curva di accuratezza durante l'apprendimento** (*learning-curve AUC*, non ROC AUC). La media avviene prima per seed, poi fra seed.

## Risultati aggregati (compiti 1–7)

| Trattamento | AUC apprendimento | Accuratezza finale | Guadagno durante il compito | Perdita media su compiti passati |
|---|---:|---:|---:|---:|
| Continua | **70,52%** | 81,69% | +31,08 pp | 32,35 pp |
| Nuova a ogni compito | **75,10%** | 83,51% | +33,98 pp | 37,83 pp* |
| Rinnovo unità poco attive | **72,87%** | 83,95% | +34,33 pp | 35,10 pp |
| Rinnovo casuale | **71,99%** | 83,15% | +33,93 pp | 34,19 pp |

\*La perdita su compiti precedenti del braccio **fresh** non è direttamente confrontabile con quella dei modelli persistenti, poiché il suo riavvio cancella volontariamente le conoscenze precedenti.

### Differenze appaiate nella metrica primaria

| Confronto AUC | Differenza media | Intervallo percentile bootstrap al 95% |
|---|---:|---:|
| Continua − Nuova | **−4,59 pp** | −5,90 / −3,18 pp |
| Rinnovo poco attive − Nuova | **−2,23 pp** | −3,46 / −0,89 pp |
| Rinnovo casuale − Nuova | **−3,12 pp** | −4,23 / −1,98 pp |
| Rinnovo poco attive − Casuale | **+0,89 pp** | +0,09 / +1,60 pp |

L'intervallo è stato calcolato con **2.000 ricampionamenti appaiati a livello di seed**. È descrittivo ed esplorativo. Non va trasformato in una pretesa di significatività universale, specialmente con vari confronti contemporanei.

L'accuratezza finale del modello a rinnovo selettivo (83,95%) è addirittura leggermente superiore a quella della rete nuova (83,51%), ma la sua AUC inferiore (72,87% contro 75,10%) indica che **impara più lentamente nel corso dello stesso compito**. Precisione finale e rapidità di apprendimento sono quindi domande distinte.

## Evoluzione per compito: AUC media fra 24 seed

| Compito | Famiglia | Continua | Nuova | Rinnovo selettivo | Rinnovo casuale |
|---:|---|---:|---:|---:|---:|
| 0 | Linea | 86,52% | 86,52% | 86,52% | 86,52% |
| 1 | XOR | 66,61% | 73,66% | 69,11% | 69,18% |
| 2 | Cerchio | 69,25% | 71,31% | 69,80% | 69,78% |
| 3 | Strisce | 70,50% | 74,53% | 71,88% | 71,69% |
| 4 | Linea | 84,46% | 86,72% | 86,44% | 84,65% |
| 5 | XOR | 65,74% | 74,36% | 69,71% | 67,64% |
| 6 | Cerchio | 70,96% | 72,13% | 71,22% | 70,48% |
| 7 | Strisce | 66,08% | 73,00% | 71,93% | 70,48% |

### Il problema dell'interpretazione

Il divario **continua − nuova** è negativo in quasi tutti i compiti successivi, ma **non cresce regolarmente** con l'età della rete. Varia invece nettamente con la famiglia di classificazione: XOR e strisce risultano più difficili per il modello continuo.

Di conseguenza, il dato **non permette ancora di distinguere** una vera perdita *cumulativa* di plasticità da un semplice effetto di **interferenza tra compiti e trasferimento negativo**. Per la domanda iniziale questa è una distinzione decisiva.

### Plasticità e dimenticanza non coincidono

Conservare tutti i pesi tutela un po' meglio le vecchie competenze nel nostro indicatore di forgetting, ma penalizza l'apprendimento dei compiti nuovi. Il rinnovo selettivo riduce lo svantaggio sui nuovi compiti, a prezzo di una perdita media superiore sui precedenti (35,10 pp contro 32,35 pp).

Le unità ReLU totalmente inattive sono rarissime in tutti i trattamenti (meno dello 0,3% medio). Il vantaggio del rinnovo selettivo **non sembra qui attribuibile semplicemente alla riattivazione di neuroni completamente morti**.

## Conclusioni ammissibili

1. Su questi problemi, con lo stesso numero di aggiornamenti e la stessa capacità nominale, **partire con pesi nuovi permette in media di apprendere più rapidamente** rispetto al mantenimento dell'intera storia.
2. Rinnovare una parte delle unità migliora l'AUC rispetto a non rinnovarle; **scegliere quelle meno attive** offre un vantaggio piccolo rispetto a un reset casuale equiparato.
3. Il vantaggio del reset comporta un **trade-off di retention** e non dimostra da solo un'intelligenza capace di autoregolare il proprio apprendimento.
4. Non abbiamo evidenza che il divario sia progressivo nel tempo, inevitabile o valido fuori dalle quattro famiglie di task sintetici.

## Esperimento successivo raccomandato: M1-P2 (conferma)

**Prima di alterare l'algoritmo**, preregistrare una sequenza più lunga su **nuovi seed mai usati** (ad esempio 300–323), mantenendo architettura e rate invariati. Usare almeno 16 compiti e confrontare esplicitamente le famiglie uguali in fasi precoci e tardive (compiti 4–7 contro 12–15). Separare **interferenza istantanea** e **degrado cumulativo**.

La conferma deve verificare:
- se il gap con una rete nuova si amplia nel tempo, a parità di tipo di compito;
- se il reset selettivo continua a battere quello casuale su nuove sequenze;
- quanta memoria dei vecchi compiti viene sacrificata;
- costo per esempio di addestramento, non soltanto precisione.

Solo dopo avrebbe senso una M2 con una **regola di rinnovo appresa ed ereditabile**, anziché fissata dall'esterno.

## Provenienza

Artefatto: [GitHub Actions #44818836611](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44818836611) → `plasticity-m1p-24-seeds`.

Contiene:
- `measurements.csv`: 24 seed × 8 compiti × 4 bracci = 768 osservazioni.
- `summary.json`: aggregati e intervalli di ricampionamento.
- `curves.svg`: andamento medio della metrica primaria per compito.

I file nel workflow hanno conservazione limitata a 30 giorni. Questo rapporto conserva le principali metriche durevolmente su GitHub.
