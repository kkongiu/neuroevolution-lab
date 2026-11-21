# M3.3 — Memoria contestuale, replay e riorganizzazione neurale

**Campagna del 21 novembre 2026** · **Esito primario: positivo, entro i limiti di un contesto esplicitamente fornito.**  
**Replay: miglioramento significativo solo nei compiti familiari.**  
**Rinnovo neurale adattivo: vantaggio non dimostrato.**

- [Protocollo M3.3 congelato prima di sviluppare e testare](../docs/M3_3_PROTOCOL.md)
- [GitHub Actions #44900094399](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44900094399), completata, test tutti superati
- [Codice M3.3](../neuroevolution/m3_3.py), [PR #13](https://github.com/kkongiu/neuroevolution-lab/pull/13) integrata su main
- [Archivio CSV per seed](M3_3_seeds_2026-11-21.csv)
- **Artefatto originale** \`m3-3-contextual-memory\`, ID **14705248735**, scadenza **21 dicembre 2026**. Contiene summary.json, seed_metrics.csv e episode_metrics.csv.

## 1. Perché M3.3 è diverso

Nella M3.2 potevamo chiedere alla stessa rete di attribuire due etichette opposte allo stesso punto x,y **senza darle modo di sapere quale regola fosse applicabile**. M3.3 rimuove questa impossibilità logica introducendo un **ID di contesto esplicito**, rappresentato con quattro ingressi one-hot aggiunti a x,y.

La rete vede quattro contesti. I primi due sono classificazioni lineari opposte per gli stessi punti; gli altri due sono XOR e il suo opposto. I *pannelli di valutazione* di ogni coppia condividono **identici punti x,y** con etichette complementari.

Una rete **senza il segnale di contesto**, anche se avesse capacità illimitata, non può raggiungere più del **50%** della media bilanciata in questo test. Questo limite deriva matematicamente dal disegno dell'esperimento, **non** è un risultato sorprendente sull'intelligenza delle reti. Il controllo cieco mantiene comunque lo stesso numero di pesi nominali, la stessa inizializzazione e lo stesso budget SGD.

La rete con contesto deve davvero imparare a usarlo: sapere che il problema è logicamente risolvibile non garantisce che una piccola rete riesca a raggiungere l'accuratezza perfetta.

## 2. Protocollo preregistrato

- Quattro contesti, 12 episodi consecutivi per seed. Ogni contesto appare inizialmente una volta, in ordine casuale; poi gli episodi richiamano contesti precedenti.
- Una sola rete persistente: MLP a **20 unità ReLU**, con **6 input** (x, y, 4 bit di contesto), addestrata durante tutti gli episodi.
- **64 aggiornamenti SGD × batch 4** per episodio; learning rate **0,04**. Pannello holdout di **64 punti per contesto** con campionamento accoppiato fra contesti opposti.
- Replay: conservazione di **24 esempi di training per contesto**; **un aggiornamento SGD su quattro** usa esempi passati di contesti diversi da quello attuale, **sostituendo** il batch corrente. Stesso numero totale di aggiornamenti, ma meno esempi del compito attuale.
- Tre controller evoluti indipendentemente: rete osservata (accuratezza, diagnostica training, attività neuronale, tempo), sola accuratezza/tempo, solo tempo. Ognuno può pianificare **tre rinnovi di tre neuroni** fra 11 transizioni.
- Baseline a parità esatta di rinnovi: intervalli uniformi, tempistiche rimescolate, stessi interventi con neuroni scelti casualmente.
- Meta-training **4 seed 20000–20003**, meta-validation **2 seed 21000–21001**, test finale principale **24 seed 22000–22023**, test con famiglie differenti **12 seed 23000–23011**.
- Le due famiglie fuori distribuzione sono strisce e scacchiera, ciascuna con regola invertita. Il controller non viene riaddestrato sui seed finali.

## 3. Risultato principale: il contesto rende realizzabile la memoria

**Accuratezza finale media, bilanciata sui quattro contesti:**

| Strategia di base | Compiti familiari — 24 seed | Compiti differenti — 12 seed |
|---|---:|---:|
| **Contesto + replay, nessun rinnovo** | **74,33%** | **62,43%** |
| **Contesto nascosto + replay, nessun rinnovo** | **50,00%** | **50,00%** |
| Contesto + nessun replay, nessun rinnovo | **69,30%** | **63,05%** |

Confronto primario preregistrato sui 24 seed principali:

- **Contesto + replay − contesto nascosto + replay: +24,33 punti percentuali**.
- Bootstrap percentile descrittivo 95%, ricampionando **24 seed**, non esempi o episodi: **+22,23 / +26,45 pp**.
- **24 seed positivi su 24**.
- La baseline senza contesto ha esattamente 50% su tutti i seed **per costruzione matematica**, non per mancanza di addestramento.

Il contesto aiuta anche nella nuova distribuzione: **+12,43 pp** rispetto al controllo cieco, intervallo bootstrap **+8,59 / +15,98 pp**, con 11 seed positivi e un pareggio. Ma non permette apprendimento perfetto: i quattro contesti restano impegnativi per questa piccola rete.

**Memoria del caso peggiore:** nel test principale, l'accuratezza del **peggiore fra i quattro contesti** è in media **52,80%** con contesto e replay: c'è quindi ancora un evidente problema di ritenzione, anche quando la media raggiunge il 74,33%.

## 4. Replay: migliora la conservazione, ma ha un costo locale

Sui **24 seed familiari**, confronto predefinito fra la *stessa rete con contesto* senza rinnovo:

| Metrica | Replay | Nessun replay | Differenza replay − senza |
|---|---:|---:|---:|
| Accuratezza finale bilanciata sui 4 contesti | **74,33%** | 69,30% | **+5,03 pp** |
| Accuratezza finale del contesto peggiore | **52,80%** | 43,49% | **+9,31 pp** |
| Dimenticanza media dal picco storico (quattro contesti) | **10,73 pp** | 16,60 pp | **−5,88 pp** |
| AUC durante i singoli episodi | **72,02%** | 73,10% | **−1,07 pp** |

Confronto preregistrato sulla memoria finale: **+5,03 pp**, intervallo bootstrap 95% **+2,07 / +8,06 pp**; **18 seed positivi, 5 negativi, 1 pari**.

Questo è il primo risultato pratico che risponde al problema M3.2: **allenarsi periodicamente anche su esperienze già viste può conservare meglio più conoscenze coesistenti**. Non è però una vittoria senza compromessi: un quarto dei batch è sottratto al contesto corrente, e infatti la sua AUC di apprendimento cala.

Il vantaggio **non si trasferisce** con evidenza sufficiente ai problemi a strisce/scacchiera: replay **62,43%** contro nessun replay **63,05%** di memoria finale, differenza **−0,62 pp**, intervallo bootstrap **−2,70 / +1,30 pp** (6 seed positivi, 6 negativi). Su questo test la protezione della memoria tramite replay è **non confermata**.

## 5. Rinnovo neurale adattivo: ipotesi non confermata

Tutti questi bracci usano contesto e replay; i sei controlli di scheduling ricevono **tre rinnovi identici per numero**, equivalenti a nove sostituzioni di neuroni nascoste nel totale.

| Strategia | AUC media episodi, familiari | Accuratezza finale bilanciata | AUC sui compiti differenti |
|---|---:|---:|---:|
| **State-aware, politica evoluta** | **71,50%** | 70,51% | 60,09% |
| Accuracy-only, politica evoluta | **71,47%** | 72,01% | 60,94% |
| Time-only, politica evoluta | **71,47%** | 72,01% | 60,94% |
| Uniforme, stessa dose | 70,78% | 68,70% | 60,36% |
| Rimescolata, stessa dose | 71,11% | 71,96% | 60,36% |
| Neuroni scelti casualmente, stesso tempismo | 70,74% | 70,80% | 60,28% |
| **Nessun rinnovo**, contesto e replay (diagnostico, non dose-matched) | **72,02%** | **74,33%** | 60,65% |

Confronto adattivo − time-only sulla metrica AUC nei 24 seed: **+0,03 pp**, intervallo bootstrap 95% **−1,46 / +1,33 pp**, 14 positivi, 9 negativi, 1 pari. Contro l'accuracy-only è **identico**. Il vantaggio sullo shuffle è soltanto **+0,39 pp** (bootstrap da −0,59 a +1,43 pp).

**Quindi l'utilità del rinnovo neurale adattivo non è dimostrata.** Anzi, sul test familiare la rete con contesto e replay **senza rinnovo** conserva meglio i quattro contesti in media (74,33%) della rete con scheduling state-aware (70,51%), pur utilizzando budget diversi.

Nel test con famiglie differenti la policy state-aware ottiene AUC **60,09%** contro **60,94%** del time-only, differenza **−0,85 pp** con intervallo da −1,79 a +0,16 pp: nessun vantaggio confermato.

Le informazioni interne **modificano alcune scelte** (32 richieste sensibili allo stato su 264 opportunità nei seed principali), ma **42 decisioni effettive** sono anche modificate dal vincolo di completare la quota. La sola variabilità comportamentale **non** rappresenta un guadagno di apprendimento.

Le policy accuracy-only e time-only selezionate hanno, entrambe, coefficienti \`[3,0,0,0,0]\`: scelgono banalmente di spendere il budget nelle prime tre transizioni. Questo rende ancora più importante non presentare il risultato state-aware come un successo.

## 6. Audit e limiti scientifici

1. **Contesto noto esternamente:** la rete non scopre quale mondo sta affrontando, lo riceve come input supervisionato. La possibilità teorica di superare il 50% è quindi incorporata nel problema.
2. **Soglia cieca del 50%:** conseguenza logica del benchmark. Il confronto positivo contesto vs cieco valida il disegno e l'apprendimento di un segnale utile, **non** una scoperta su forme di intelligenza generale.
3. **Replay riduce i batch del compito attuale:** l'esperimento pareggia il lavoro SGD, non l'esposizione ai dati del compito corrente; il guadagno di memoria va interpretato assieme alla perdita di AUC corrente.
4. **Precisione non perfetta:** il contesto peggiore della rete con replay è vicino al 50% nel test familiare; la coesistenza logica non garantisce la capacità pratica della rete o un recupero affidabile della memoria.
5. **Meta-apprendimento limitato:** le policy numeriche sono evolute da un algoritmo esterno. Non cambiano regole SGD, codice o ricompense.
6. **Deviazione limitata dal protocollo nella diagnostica del controller:** il protocollo descrive l'accuratezza sugli esempi conservati degli *altri contesti*. Nell'implementazione il valore osservato include **tutti i contesti precedentemente archiviati**, anche quello corrente quando ritorna. Non riguarda i due confronti cue/replay **senza rinnovo** ma limita l'interpretazione del controller state-aware. Non modifichiamo retroattivamente il codice della run; un eventuale fix richiederà test nuovi.
7. **Seed e holdout separati:** training, validazione e i due test finali hanno range indipendenti. Le statistiche usano intervalli descrittivi appaiati per seed; l'unità sperimentale non è la singola immagine, etichetta o episodio.
8. **Famiglie OOD specifiche:** XOR/line contro checker/stripe sono un cambio parziale dei tipi di problemi; le regole restano 2D e gli indizi di contesto mantengono esattamente la stessa forma.

## 7. Decisione e prossimo gate

**Confermata in questo benchmark la possibilità di mantenere più rappresentazioni reciprocamente incompatibili grazie a un segnale di contesto esplicito.** Replay migliora la memoria sui task familiari a costo di parte dell'apprendimento immediato. Non abbiamo dimostrato che tale beneficio si estenda alle famiglie nuove, né che la schedulazione neurale adattiva migliori una politica temporale.

**Proposta M3.4:** smettere di forzare tre rinnovi per ogni rete e ottimizzare **solo il costo-opportunità** di memoria, capacità e riorganizzazione. Aggiungere inferenza del contesto dai dati senza un ID fornito, memoria episodica esplicita con ablation di replay, e confronto equo a budget totale di operazioni. Scegliere *prima* nuovi seed e un test di trasferimento più severo. Non reinterpretare né modificare M3.3 alla luce dei risultati.

### Provenienza

L'artefatto [GitHub Actions #44900094399](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44900094399) contiene:
- **summary.json**: risultati aggregati e bootstrap per seed, coefficienti scelti, limiti.
- **seed_metrics.csv**: 36 seed × 9 bracci = **324** esecuzioni.
- **episode_metrics.csv**: 36 × 9 × 12 = **3.888** osservazioni episodio/braccio.

I dati per seed più importanti sono archiviati stabilmente in [M3_3_seeds_2026-11-21.csv](M3_3_seeds_2026-11-21.csv); i dati completi originali sono in GitHub Actions con retention di 30 giorni.
