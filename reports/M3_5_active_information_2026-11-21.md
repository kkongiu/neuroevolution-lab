# M3.5 — Il valore dell'informazione: chiedere quando il contesto è incerto

**Campagna:** 21 novembre 2026. **Risultato primario NON confermato:** l'accuratezza media con richieste guidate dall'incertezza supera numericamente le richieste casuali **a parità esatta di informazioni**, ma l'intervallo bootstrap descrittivo include zero.

- [Protocollo preregistrato](../docs/M3_5_PROTOCOL.md) scritto prima del codice e dei test finali.
- [Codice M3.5](../neuroevolution/m3_5.py), [test](../tests/test_m3_5.py), [PR #17](https://github.com/kkongiu/neuroevolution-lab/pull/17) integrata in main.
- [GitHub Actions #44903487190](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44903487190), **success**, campagna completa con test superati.
- [Archivio CSV per ciascuno dei 48 seed](M3_5_seed_accuracy_2026-11-21.csv).
- **Artefatto originale**: GitHub Actions `m3-5-active-information` ID **14707330313**, scadenza prevista **21 dicembre 2026**. Contiene `summary.json`, `seed_metrics.csv` (336 righe), `request_metrics.csv` (3456 righe di decisioni).

## 1. La domanda scientifica

In M3.4 il riconoscitore aveva dimostrato capacità di richiamare contesti con sedici indizi corretti, ma peggiorava con soli sei indizi di cui uno sbagliato. Può **riconoscere un momento di incertezza e spendere lì una quantità fissa di feedback aggiuntivo**, migliorando rispetto a richieste *casuali o periodiche con lo stesso costo*?

Il sistema non scopre autonomamente il metodo: un ricercatore esterno ha implementato un algoritmo kNN con una soglia di incertezza, e un algoritmo di ricerca esterno sceglie la soglia usando soltanto meta-training e meta-validation. Non è autodeterminazione.

## 2. Protocollo e controlli

- **4 contesti latenti**, presentati durante **12 episodi**; regole base line/XOR (o stripe/checker nel trasferimento) con coppie di etichette opposte su identici input x,y.
- Per ogni episodio, tutte le strategie ricevono **6 esempi di supporto etichettati, 1 volutamente errato**.
- Le quattro strategie a budget identico possono richiedere nuovi dati in **esattamente 4 episodi su 12**, ricevendo **2 esempi nuovi per richiesta**: **8 etichette aggiuntive totali** per seed. La richiesta è decisa **prima** di leggere le etichette aggiuntive.
- Strategie: *margin uncertainty* (distanza fra punteggi delle corrispondenze), *entropy uncertainty* (entropia delle corrispondenze), richieste casuali, richieste periodiche. Altri controlli, **non dose-matched**: nessuna richiesta oppure richiesta in tutti i 12 episodi (24 etichette extra).
- **Rete persistente** con 24 ReLU, 6 input: x,y e quattro bit di slot dedotto. Addestramento uguale in tutti i bracci: **48 SGD update × batch 4 per episodio**, replay FIFO di TRAINING sostituito ogni quarto batch, capienza 24 esempi per slot. Prototipi separati per riconoscimento: 64 esempi TRAINING per slot, kNN k=7.
- Il vero ID del contesto e le etichette dei pannelli HOLDOUT sono usati solo per generare il mondo e per l'audit post-hoc, **non per decidere se chiedere**, assegnare slot o fare replay.
- **Soglia del resolver 0,84** invariata da M3.4. Scelta meta-training/meta-validation delle nuove soglie di richiesta: **0,70 per margine**, **0,75 per entropia**.
- Seed separati: meta-train **29000–29005**, meta-validation **29500–29502**, main indipendente **30000–30023** (24 seed), famiglie nuove **31000–31011** (12), feedback aggiuntivo con un'altra etichetta errata **32000–32011** (12).

**Metrica principale congelata:** accuratezza finale media bilanciata sui quattro contesti, *margine meno random*, accoppiata per seed, con uguale budget di 8 etichette supplementari.

La metrica è accuratezza finale, non learning-curve AUC né ROC AUC. La curva di apprendimento e la dimenticanza sono diagnostiche secondarie.

## 3. Test principale: 24 seed e informazioni aggiuntive pulite

| Strategia | Accuratezza finale bilanciata | Coerenza ritorni | Accuratezza contesto peggiore | Dimenticanza dal picco |
|---|---:|---:|---:|---:|
| **Richiesta per incertezza del margine** | **65,30%** | **77,08%** | 38,74% | 16,76 pp |
| Richiesta per entropia | 64,31% | 70,83% | 37,24% | 18,07 pp |
| Richiesta casuale, stessa dose | 62,92% | 64,06% | 36,59% | 19,04 pp |
| Richiesta periodica, stessa dose | 61,08% | 58,85% | 32,81% | 20,49 pp |
| **Nessuna richiesta extra**, dose 0 | 61,02% | 61,46% | 33,53% | 20,90 pp |
| Richiesta in ogni episodio, dose tripla | 65,64% | 76,56% | 38,67% | 16,52 pp |
| ID di contesto privilegiato (riferimento non deployable) | 73,40% | *non confrontabile* | 47,92% | 9,94 pp |

### Confronti appaiati prespecificati

| Margine meno... | Vantaggio medio sulla memoria finale | Bootstrap descrittivo 95% | Seed + / − / = |
|---|---:|---:|---|
| **Casuale — PRIMARIO** | **+2,38 punti** | **−1,56 / +7,27** | **7 / 4 / 13** |
| **Periodico — secondario necessario** | **+4,22 punti** | **+0,68 / +8,61** | **11 / 3 / 10** |
| Nessuna richiesta extra | +4,28 punti | +0,85 / +8,50 | 11 / 2 / 11 |
| Richiesta in tutti gli episodi | −0,34 punti | −3,52 / +2,78 | 3 / 7 / 14 |
| Entropia, stessa dose | +0,99 punti | −1,34 / +3,56 | 6 / 4 / 14 |

**Conclusione preregistrata: la superiorità sulla richiesta CASUALE non è dimostrata**, perché il confronto primario ha un intervallo che include zero. Il miglioramento rispetto alla pianificazione periodica è favorevole, ma **non sostituisce** la verifica primaria fallita.

Il vantaggio medio sul casuale, diviso per lo stesso budget di otto etichette, è **+0,297 punti percentuali per etichetta aggiuntiva** come rapporto descrittivo. **Non** è una stima causale del valore assoluto di un singolo esempio: entrambi i bracci ricevono otto etichette aggiuntive.

L'inferenza sembra qualitativamente migliore con il margine: coerenza dei ritorni **77,08%** vs **64,06%** random e qualità media delle coppie di contesti **84,57%** vs **76,69%**. **Non** inferiamo automaticamente significatività statistica da queste differenze diagnostiche.

## 4. Ha davvero deciso autonomamente quando chiedere?

È necessario distinguere le richieste originate dall'incertezza da quelle imposte dal budget.

Sul test principale (24 × 12 = 288 opportunità):
- Il controller a margine ha effettuato **96 richieste** (4 per seed), tutte richieste dalla propria soglia, **nessuna imposta dalla quota residua**.
- Dopo avere speso il budget, ha prodotto altre **72 intenzioni di richiesta che il limite ha bloccato**. Questi blocchi NON sono richieste eseguite né decisioni autonomamente evitate.
- Il controller a entropia ha effettuato **96 richieste**, delle quali **65** effettivamente richieste in base alla soglia e **31 forzate** dal meccanismo di spesa della quota; una richiesta addizionale è stata bloccata dopo l'esaurimento.
- Casualità e periodicità spendono entrambe esattamente 96 richieste totali, con la stessa esposizione di 192 etichette extra aggregate (8 per seed).

La capacità di indirizzare le richieste in base alla situazione è dunque **operativamente presente** nel controller a margine, ma resta una *soglia costruita e selezionata esternamente*. Non è la rete neurale che inventa uno scopo o una regola di esplorazione.

Una parte delle richieste casuali cade su episodi molto precoci, in cui mancano ancora prototipi affidabili; questa differenza di opportunità può contribuire al divario osservato. L'alternativa periodica è un controllo ulteriore per questo confondente, ma la superiorità numerica del margine resta specifica al mondo studiato.

## 5. Trasferimento: famiglie diverse e feedback inaffidabile

| Strategia | Line/XOR noto, 24 seed | Stripe/checker nuovi, 12 seed | Line/XOR ma anche 1 errore nelle 2 etichette extra, 12 seed |
|---|---:|---:|---:|
| **Margine/incertezza** | **65,30%** | **56,32%** | **63,74%** |
| Entropia | 64,31% | 56,25% | 60,64% |
| Casuale, stessa dose | 62,92% | 54,04% | 61,13% |
| Periodico, stessa dose | 61,08% | **57,16%** | 63,25% |
| Nessuna richiesta extra | 61,02% | **57,16%** | **63,87%** |
| Richiesta in tutti gli episodi (dose tripla) | 65,64% | 53,26% | 60,25% |
| Contesto reale privilegiato | 73,40% | 61,04% | 70,74% |

Confronto **margine − casuale**, a pari dose:
- **Nuove famiglie:** +2,28 pp, intervallo descrittivo bootstrap **−0,20 / +5,05 pp**; 5 seed positivi, 1 negativo, 6 pari. **Non confermato**.
- **Feedback extra corrotto:** +2,60 pp, intervallo **+0,03 / +6,15 pp**; 4 seed positivi, 2 negativi, **6 pari**. È soltanto un segnale esplorativo fragile su 12 seed, non una conferma robusta della generalizzazione del vantaggio primario.

Nonostante il confronto sul casuale, nelle **nuove famiglie** il margine **perde 0,85 pp contro la strategia che non chiede niente**. Con **extra feedback errato**, il margine ottiene **63,74%** contro **63,87% senza richieste extra**, e richiedere in ogni episodio peggiora fino al **60,25%**. **Più feedback non garantisce informazioni migliori.**

La coerenza del riconoscimento dei contesti che ritornano con il margine passa da **77,08%** nel test principale a **52,08%** sulle nuove famiglie e **55,21%** con extra feedback corrotto. Questa fragilità è il principale limite da affrontare.

## 6. Risorse, correttezza e limiti

- Tutti i bracci a parità di dose hanno **4 richieste × 2 esempi = 8 etichette extra per seed**, oltre a 72 etichette iniziali; i bracci con nessuna richiesta e richiesta sempre sono fuori budget e soltanto diagnostici.
- I bracci ricevono gli stessi **48 × 4 SGD update/example per episodio**, la stessa rete iniziale, gli stessi pannelli heldout e la stessa capienza massima di replay (96 esempi) e prototipi (256 esempi).
- La metrica di costo in etichette è equa ma **non implica tempo CPU identico**: sul test principale il conteggio medio delle valutazioni kNN è circa **4698 per seed** per margine contro **4669 random**, **4704 periodico** e **4515 senza richiesta**; le decisioni influenzano le successive tracce di memoria.
- Dopo che uno slot viene assegnato diversamente, anche i gruppi usati per il replay possono divergere: è un effetto conseguente all'inferenza, non un'ulteriore scelta parametrica nel confronto.
- Alcuni seed danno lo stesso punteggio per politiche diverse: nel primario **13/24 sono pareggi**. Non basta scegliere il miglior valore medio o osservare una curva crescente.
- Il test main ha 24 repliche indipendenti; i due test di trasferimento ne hanno 12 ciascuno. Bootstrap descrittivo appaiato a livello di **seed**, non per singolo punto, richiesta o episodio; i molti confronti secondari non danno evidenza universale.
- Gli ID latenti e i pannelli holdout non sono accessibili al runtime, mentre l'**offline meta-selection** usa i veri ID **solo** sulle sequenze train/validation per valutare i cluster. Questo non prova inferenza non supervisionata.
- L'algoritmo non sceglie *quali x,y* etichettare, ma solo *quando* chiedere 2 esempi IID. Non è ancora active learning con scelta ottimizzata dei campioni.
- Il riferimento con ID privilegiato conserva un'accuratezza generalmente più alta, ma non è una policy deployable né un limite superiore garantito per ogni rete iniziale.
- L'implementazione originale registra `forced_or_blocked_count` **insieme**; per distinguere i casi, dal file `request_metrics.csv` è sufficiente incrociare i campi `query` / `requested` / `forced_or_blocked`. I numeri riportati qui sono stati ricavati da tale dataset senza alterare la campagna congelata.

## 7. Decisione e ipotesi successiva

**M3.5 NON conferma ancora il beneficio universale del «chiedere quando sono incerto»**, anche se il meccanismo reagisce realmente all'incertezza e in ambienti familiari ottiene vantaggi rispetto a una strategia periodica o nessuna richiesta. Sul random, che è il confronto principale, l'intervallo include zero; su famiglie nuove o feedback errato l'utilità è poco affidabile.

Una M3.6 sensata dovrebbe **separare il valore e l'affidabilità delle informazioni** dal semplice tempismo: la policy dovrebbe poter rifiutare una risposta sospetta o chiedere conferma di un campione e scegliere un punto x,y in modo informativo, anziché chiedere sempre due osservazioni IID. Andrebbero poi implementati test con feedback rumoroso in modo stocastico e almeno un *costo esplicito* dell'interrogazione, confrontando con una policy che può **spendere meno del budget massimo**.

Tutto deve essere preregistrato su nuovi seed, con una valutazione di calibrazione dell'incertezza e un confronto fair contro strategie semplici; **non** reinterpretare a posteriori M3.5 né riutilizzare i seed finali per selezionare la nuova policy.

### Dati

L'originale [GitHub Actions #44903487190](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44903487190) contiene `summary.json`, `seed_metrics.csv` (48 × 7 = 336 righe) e `request_metrics.csv` (48 × 6 × 12 = 3456 righe di query).

L'archivio compatto [M3_5_seed_accuracy_2026-11-21.csv](M3_5_seed_accuracy_2026-11-21.csv) conserva i punteggi finali dei 48 seed e dei sette controlli per confronti futuri anche dopo la scadenza dell'artefatto Actions.
