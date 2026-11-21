# M4.0 — Primo mondo di sopravvivenza: evoluzione neurale, cibo e ostacoli

**Data campagna:** 21 novembre 2026. **Esito preregistrato: NEGATIVO sul confronto primario.**

**In sintesi:** è stato costruito e validato un mondo 2D nel quale agenti con rete neurale ricorrente si muovono, consumano energia, incontrano muri, raccolgono cibo e muoiono. La selezione genetica produce un comportamento alimentare migliore di una **singola** rete non evoluta, ma **inferiore** persino a un agente che sceglie direzioni casuali evitando i muri. La differenza principale è **negativa con intervallo bootstrap che non comprende zero**. Il forte controllo euristico basato sull'odore è nettamente superiore.

- [Protocollo M4.0 preregistrato](../docs/M4_0_PROTOCOL.md), committed prima di qualsiasi test finale.
- [Codice simulatore ed evolutore](../neuroevolution/m4_0.py), [test](../tests/test_m4_0.py).
- [PR #19](https://github.com/kkongiu/neuroevolution-lab/pull/19) integrata in main.
- [Campagna originale GitHub Actions #44904926385](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44904926385), successo.
- [Dati permanenti: food raccolto per ciascuno dei 48 mondi](M4_0_food_by_world_2026-11-21.csv).
- Artefatto GitHub Actions **m4-0-survival-world**, ID **14707716591** (scadenza prevista **21 dicembre 2026**): contiene summary.json, champion.json, history.csv, test_seed_metrics.csv e due mappe SVG con tracce dei movimenti e JSON passo-passo.

## 1. Descrizione concreta dell'ecosistema

- Mondo 2D **13 × 13** con bordo invalicabile, muri interni casuali e **cibo solo in celle raggiungibili**, verificato per ogni seed.
- L'organismo inizia al centro con **24 unità di energia**, ne consuma almeno 1 per mossa e ulteriori 0,3 in caso di collisione; una risorsa alimentare aggiunge fino a 16 unità di energia, con limite 42. Esaurire l'energia produce la morte simulata.
- **Orizzonte massimo 100 mosse**, 12 unità di cibo iniziali per mondo; ciascuna può essere consumata una volta sola e non ricresce.
- Sensori numerici (15): ostacoli immediatamente adiacenti, cibo adiacente, **odore direzionale entro cinque celle**, energia e feedback immediato di urto/nutrimento. Niente mappa completa o percorso ottimo.
- Rete **ricorrente** (12 unità tanh, 388 parametri evolutivi) che seleziona una delle quattro direzioni cardinali. Gli stati ricorrenti aggiornano la memoria a breve termine **durante la vita**; i pesi non vengono appresi o modificati durante la vita.
- Un algoritmo esterno usa **12 genomi × 8 generazioni × 8 mondi di training = 768** prove training; 4 mondi di validazione indipendenti per ogni leader generazionale (32 valutazioni validation). Il singolo campione genetico scelto è il leader della **generazione 2**, selezionato solo sulla validazione. Nessun accesso ai seed finali nella scelta.
- Mondi di training 41000–41007; validazione 42000–42003; test main 43000–43031; stress test **44000–44015** (muri più densi: 0,26, solo 8 cibi e odore entro due celle).
- Controlli: rete neurale casuale **non evoluta** con stessa architettura; camminatore casuale che evita gli ostacoli; algoritmo esplicito che segue il massimo odore evitando gli ostacoli; rete evoluta con **solo i quattro canali dell'odore azzerati**, per testarne il ruolo causale.

## 2. Risultati scientifici — 32 mondi mai visti

| Strategia | Cibo medio raccolto (su 12) | Passi di sopravvivenza | Urti medi | Frazione morta per fame |
|---|---:|---:|---:|---:|
| **Rete ricorrente evoluta** | **2,19** | **47,53** | **17,66** | **100%** |
| Rete ricorrente non evoluta (un solo genoma casuale) | 0,56 | 31,63 | 1,22 | 100% |
| **Casuale con prevenzione collisioni** | **3,69** | **70,75** | **0** | 75% |
| **Regola euristica: massimo odore, evitare muri** | **8,47** | 70,69 | **0** | 93,75% |
| Stessa rete evoluta senza percezione dell'odore | 2,59 | 49,56 | 18,66 | 100% |

### Contrasti appaiati per mondo

| Rete evoluta meno... | Δ cibo medio raccolto | Intervallo bootstrap descrittivo 95% | Mondi: + / − / = |
|---|---:|---:|---:|
| **Casuale senza collisioni (primario)** | **−1,50** | **−2,53 / −0,47** | 10 / 20 / 2 |
| Rete non evoluta (secondario) | +1,63 | +0,94 / +2,41 | 21 / 3 / 8 |
| Algoritmo esplicito di inseguimento dell'odore | **−6,28** | **−7,59 / −4,88** | 2 / 29 / 1 |
| Stessa rete con odore disabilitato | −0,41 | −1,25 / +0,47 | 11 / 15 / 6 |

Il test primario fissato prima dei risultati è quindi **fallito**, con evidenza di un *peggioramento* rispetto a una strategia elementare che non urta gli ostacoli. La rete ha comunque raccolto almeno un cibo in **26/32 mondi (81,25%)** e supera l'unico RNN casuale confrontato. Questi segnali non bastano a confermare il successo perché l'esperimento richiedeva un vantaggio sul camminatore con evitamento delle collisioni.

**L'ablation di odore non prova l'utilizzo utile dell'odore**: l'evoluta con odore disattivato raccoglie addirittura 0,41 cibi in più in media; l'intervallo è compatibile con zero. La rete può star sfruttando comportamenti di movimento accidentali anziché seguire il segnale alimentare.

### Un esempio concreto del problema

Nella mappa test 43000 il campione evoluto raccoglie inizialmente **due risorse**, poi si incastra ripetutamente presso una parete: **24 collisioni su 42 mosse, soltanto 8 celle distinte visitate, morte per fame**. Questo comportamento è conservato integralmente in una mappa SVG e nel file JSON cronologico dell'artefatto.

Il problema è strutturale anche nella funzione obiettivo: ogni cibo vale **+2 punti fitness**, mentre ogni collisione costa solo **−0,05**, oltre al costo energetico. Questi pesi furono fissati prima della campagna; il sistema può quindi tollerare troppe collisioni. **Non li modifichiamo retroattivamente.**

## 3. Stress test — 16 mondi più difficili, stessa rete congelata

| Strategia | Cibo medio (su 8) | Passi di sopravvivenza | Urti medi |
|---|---:|---:|---:|
| **Evoluta** | **1,31** | **38,69** | **15,75** |
| Rete casuale non evoluta | 0,63 | 32,75 | 0,63 |
| Casuale che evita i muri | 2,00 | 53,19 | 0 |
| Greedy basato sull'odore | 2,69 | 50,44 | 0 |
| Evoluta senza odore | 1,50 | 38,06 | 16,75 |

- Evoluta − camminatore casuale con collisioni evitate: **−0,69 cibi**, bootstrap **−1,56 / +0,31**; segno medio sfavorevole e incertezza compatibile con zero.
- Evoluta − RNN non evoluta: **+0,69 cibi**, bootstrap **+0,06 / +1,44**; confronto esplorativo con **un solo** genoma casuale, non una popolazione indipendente.
- Evoluta − algoritmo greedy: **−1,38 cibi**, bootstrap **−2,38 / −0,25**.
- Rete evoluta mangia almeno una risorsa in **10/16 mondi (62,5%)**, meno dell'81,25% ottenuto sui mondi familiari. Nessun organismo evoluto raggiunge il limite delle 100 mosse senza morire di fame.

Il trasferimento al mondo più difficile **non giustifica alcuna rivendicazione di robusta navigazione autonoma**.

## 4. Evoluzione: miglioramento nel training, selezione per validazione

La fitness migliore di training è migliorata da **3,29** alla generazione 0 a **6,90** dalla generazione 3 in poi. Ma la migliore validazione è comparsa già nella **generazione 2** (**5,06**), mentre le generazioni successive hanno ottenuto una validazione di circa **3,60**, nonostante training migliore.

Questo è un segnale di possibile **adattamento eccessivo ai soli otto mondi di addestramento**. Non prova definitivamente overfitting, ma giustifica test indipendenti e rotazioni/variazioni dei mondi nella fase successiva. La selezione sulla validazione ha rispettato il protocollo senza sostituire retroattivamente il genoma con quello favorito dai dati finali.

## 5. Limiti della ricerca

1. **Una sola evoluzione completa** su otto generazioni e un solo genoma casuale di riferimento: la popolazione iniziale e i suoi risultati potrebbero cambiare con altri seed. I 32 mondi finali misurano la variabilità dell'ambiente, non quella delle *run evolutive*. Non fingere un campione di 32 evoluzioni indipendenti.
2. La rete riceve un **gradiente d'odore progettato da noi**. La regola greedy ottiene un enorme vantaggio grazie allo stesso segnale. Questo non è navigare usando visione o memoria spaziale autonoma.
3. **Controllo motore inadeguato:** per errore di decisione l'organismo tenta spesso di attraversare i muri. Un sistema che evita anche solo le collisioni lo batte. Il numero di urti è un indicatore fisicamente spiegabile della causa del fallimento.
4. Il cervello contiene ricorrenza ma **nessun apprendimento di pesi durante la vita**: l'evoluzione agisce solo fra generazioni artificiali.
5. Non esistono ancora predatori, locomozione continua, riproduzione in-world, rigenerazione del cibo, ecosistemi bilanciati o autonomia nella scelta della funzione obiettivo.
6. Migliorare sul genoma casuale fisso è un confronto debole rispetto a molti genomi non evoluti campionati indipendentemente; non viene scambiato per conferma scientifica dell'intelligenza.
7. Gli intervalli bootstrap sono descrittivi e calcolati su **world seed appaiati**, senza fingere indipendenza dei 100 step dentro lo stesso mondo. Alcuni confronti sono esplorativi, non corretti per molteplicità.

## 6. Decisione: M4.0 validato come simulatore, non come prova di navigazione evoluta efficiente

**Stato ingegneristico: positivo.** Disponiamo di mondi con fisica semplice, reti evolutive, controllo di risorse ed energia, test automatici, riproducibilità, visualizzazioni SVG e metriche archiviate. Gli organismi assumono azioni effettive nel mondo e la selezione genetica modifica la performance rispetto a una rete iniziale fissa.

**Stato scientifico: primario negativo.** Non è stato dimostrato il successo evolutivo rispetto alla semplice strategia di evitare i muri. L'agente non mostra navigazione efficace, non conserva energie a sufficienza e non sfrutta in modo convincente il segnale d'odore.

**Proposta M4.1**, da preregistrare prima dei nuovi risultati:
- Introdurre un **decoder che maschera le azioni dirette ai muri**, uguale per le reti evolve/no-evolve, per isolare la strategia alimentare dal controllo dei movimenti legali; confrontare contro una versione senza maschera e con stessa quantità di ricerca evolutiva.
- Aumentare la variabilità dei mondi mediante rotazioni, ostacoli e partenze differenti, mantenendo comparabilità statistica e nuovi seed test: valutare se si riduce l'overfitting.
- Ripetere l'evoluzione da **molti seed genetici indipendenti**, non confrontare una sola rete evoluta contro una sola rete random.
- Misurare alimentazione, collisioni, uso dell'odore, movimento in celle nuove e durata con controlli euristici forti. Solo dopo test di orientamento e memoria spaziale introdurre riproduzione e predatori.
- Nessuna delle modifiche deve ricalibrare ex post le metriche o alterare il record M4.0.

La prossima fase deve rispondere a una domanda più ristretta e falsificabile: **una volta impedito di attraversare muri, una rete evoluta sa trovare cibo meglio di una regola casuale lecita e generalizzare oltre la mappa vista?**

## Dati originali e riproduzione

[GitHub Actions #44904926385](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44904926385) produce, oltre ai test, i seguenti artefatti:
- champion.json: vettore del genoma selezionato, **388 parametri** e metadati.
- history.csv: tutte le otto generazioni, medie training e validazione.
- test_seed_metrics.csv: **240 osservazioni** (48 mondi × 5 politiche), compresi sopravvivenza, collisioni, energie e visite.
- summary.json: aggregati, dimensioni del test e differenze appaiate con intervalli.
- example_main.svg / example_shifted.svg: **mappe 2D reali**, cibo iniziale, traiettoria dell'agente, partenza e punto finale.
- example_main_trace.json / example_shifted_trace.json: registro passo-passo di posizione, energia, cibo e urti.

Per non perdere i dati principali alla scadenza degli artefatti Actions, il repository conserva questo rapporto e il [CSV per mondo](M4_0_food_by_world_2026-11-21.csv). Il codice con seed/genotipo selezionato in modo deterministico consente di ripetere l'intero esperimento anche senza il ZIP.
