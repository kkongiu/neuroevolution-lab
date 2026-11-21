# M4.2 — Memoria spaziale con indizio temporaneo: vantaggio di architettura, prova causale di memoria non confermata

**Campagna:** 21 novembre 2026. **Confronto primario fra architetture: positivo secondo il criterio preregistrato.**  
**Ipotesi scientifica più forte, memoria ricorrente utilizzata per decidere al bivio: NON CONFERMATA**, perché i necessari interventi causali sullo stato e sul segnale non superano le soglie fissate prima dell'esecuzione.

- [Protocollo M4.2 congelato prima del codice e dei seed finali](../docs/M4_2_PROTOCOL.md).
- [Codice](../neuroevolution/m4_2.py), [test](../tests/test_m4_2.py), [PR #23 integrata](https://github.com/kkongiu/neuroevolution-lab/pull/23).
- [GitHub Actions #44908425038](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44908425038): completa con successo, controlli di regressione e campagne preregistrate.
- [Dati persistenti: 16 gruppi split × seed genetico e sette strategie](M4_2_success_by_genetic_run_2026-11-21.csv).
- **Artefatto originale** `m4-2-delayed-cue-memory` (ID **14709166827**, scadenza stimata 21 dicembre 2026): contiene `summary.json`, `genotypes.json`, `train_validation.csv`, `delayed_trial_metrics.csv`, `reactive_transfer_metrics.csv`, due mappe SVG reali e tracce JSON passo-passo.

## 1. Un esperimento che richiede veramente informazione passata

A differenza delle prove precedenti di classificazione, questo studio presenta un **corridoio a T** in una griglia 13×13. Quando l'organismo parte riceve, per due sole decisioni motorie, un segnale che indica in quale direzione assoluta si trova l'unica unità di cibo. Il segnale scompare **prima** che raggiunga un bivio simmetrico. Due bracci contengono portali a senso unico: scegliere un braccio chiude l'episodio con cibo soltanto se è quello indicato inizialmente.

Per ciascun seed vengono generati **due mondi con stesse pareti e stessa partenza**, ma obiettivi opposti. Al bivio:
- il segnale è nullo su entrambi i mondi;
- non esiste odore di cibo, né percezione di cibo adiacente;
- l'energia reale diminuisce, ma il suo canale sensoriale rimane **costante a 1** per impedire di codificare il vecchio segnale con una strategia di attesa e leggere l'energia come un orologio;
- il controllo motorio impedisce l'attraversamento delle pareti per tutte le strategie.

**Il mondo è deliberatamente semplificato:** il passaggio in un braccio decide l'esito, invece di richiedere una lunga esplorazione oltre il bivio. È un test circoscritto della memoria, non un ecosistema con predatori o riproduzione.

La rete ricorrente usa 8 unità tanh e **260 pesi**; la feedforward senza memoria usa 11 unità e **268 pesi** (+3,1%, controllo conservativo). Evoluzione numerica esterna dei pesi fra le vite; **nessuna modifica sinaptica durante una vita**.

## 2. Disegno preregistrato

- **8 seed genetici indipendenti, 85000–85007**, ciascuno con evoluzione separata per RNN e feedforward, stessa procedura, stessi mondi e stesso budget.
- Per ciascuna architettura e seed: **6 generazioni ×12 genomi ×10 coppie di mondi ×2 obiettivi =1.440 episodi training**. Totale complessivo **23.040 episodi training**.
- Mondi di training 55000–55059; selezione delle otto coppie di campioni solo su validazione 56000–56005.
- Test principali mai visti: **32 coppie di mondi 57000–57031**, con lunghezza del corridoio 3–5 mosse; test su ritardi maggiori: **16 coppie 57500–57515** con corridoi 5–6 mosse.
- Trasferimento esclusivamente diagnostico, senza riaddestramento: **12 nuovi labirinti aperti 58000–58011**, con odore del cibo attivo e confronto con casuale legale, euristica che segue l'odore, rete con odore azzerato.
- Sette bracci principali: RNN, feedforward, stessa RNN con stato nascosto azzerato al primo arrivo al bivio, stessa RNN con cue iniziale invertito, stessa RNN senza cue, casuale legale e controllore a **memoria simbolica esplicita**, che ricorda l'indicazione e avanza nel corridoio.
- Tutti i risultati principali usano **bootstrap gerarchico appaiato** ricampionando separatamente interi seed genetici (8) e coppie di mondi (32), non trattando i singoli passi o entrambi i target della stessa coppia come repliche indipendenti.

## 3. Esito principale: la rete ricorrente batte la feedforward

**Successo nel scegliere il ramo corretto e ottenere il cibo**, media nel test main:

| Strategia | Successo | Arriva al bivio | Preferenze illegali corrette per episodio |
|---|---:|---:|---:|
| **Rete ricorrente evoluta** | **39,26%** | **72,07%** | 5,31 |
| **Rete feedforward evoluta** | **28,13%** | **56,25%** | 5,02 |
| Stessa RNN, memoria azzerata al bivio | 33,01% | 72,07% | 5,18 |
| Stessa RNN, cue iniziale invertito | 30,47% | 72,07% | 5,31 |
| Stessa RNN, cue assente | 27,93% | 58,98% | 6,39 |
| Casuale che sceglie solo direzioni legali | 34,38% | 75,00% | 0 |
| **Memoria programmata con una variabile** | **100,00%** | **100,00%** | 0 |

**Confronto primario preregistrato RNN − FF: +11,13 punti percentuali**, intervallo descrittivo bootstrap gerarchico al 95% **+2,34/+20,90 pp**; differenza positiva in **7 delle 8 evoluzioni genetiche indipendenti**, un pareggio, zero negative. **Il gate primario fra architetture è superato**.

Tuttavia una parte importante dell'effetto è già nel **raggiungere fisicamente il bivio**: 72,07% RNN contro 56,25% FF. Non possiamo chiamare l'intero +11,13 pp un vantaggio di *memoria del cue al momento della scelta*: potrebbe dipendere dalla maggiore efficacia del controllo del corridoio dovuta alla ricorrenza.

## 4. I due test causali necessari NON confermano la memoria

| Contrasto preregistrato | Delta successo medio | Bootstrap gerarchico 95% | Seed genetici: + / − / = |
|---|---:|---:|---|
| **RNN normale − stessa RNN con stato azzerato al bivio** | **+6,25 pp** | **−1,95 / +14,45 pp** | **5 / 3 / 0** |
| **RNN normale − stessa RNN con cue iniziale scambiato** | **+8,79 pp** | **−6,64 / +23,44 pp** | **5 / 2 / 1** |
| RNN normale − stessa RNN senza alcun cue | +11,33 pp | +2,15 / +21,48 pp | 6 / 1 / 1 |
| RNN normale − movimento casuale legale | +4,88 pp | −5,66 / +16,60 pp | 4 / 3 / 1 |
| RNN normale − controllore di memoria simbolica | **−60,74 pp** | **−69,34 / −52,34 pp** | 0 / 8 / 0 |

Azzerare i neuroni ricorrenti al bivio riduce la riuscita media, ma l'intervallo include lo zero; cambiare il segnale iniziale senza cambiare il mondo presenta lo stesso problema di incertezza. **I due criteri di causalità sulla memoria sono dunque falliti.**

L'effetto positivo del cue presente/assente è interessante, ma **non prova da solo che la rete conservi l'indicazione fino al bivio**: il cue può influenzare il modo in cui l'organismo percorre il corridoio, per esempio se riesce a raggiungere il bivio. I controller feedforward ottengono il 50% soltanto **condizionatamente** a una scelta simmetrica al bivio: il successo complessivo del 28,13% riflette anche episodi in cui non vi arrivano.

Il controllore scriptato che mantiene soltanto la direzione iniziale raggiunge **100%**, quindi il problema è risolvibile. La rete evoluta resta nettamente al di sotto di questa strategia semplicissima.

**Conclusione corretta:** la ricorrenza dà un vantaggio architetturale su questi mondi, ma **la presenza di una memoria neurale causalmente utile e robusta per la scelta del ramo non è ancora dimostrata**.

## 5. Ritardo più lungo: nessuna generalizzazione confermata

Su **16 coppie di mondi mai viste** con corridoio più lungo e stesso modello congelato:

| Strategia | Successo su obiettivo |
|---|---:|
| RNN evoluta | **26,17%** |
| Feedforward evoluta | 25,00% |
| RNN stato azzerato al bivio | 21,88% |
| RNN cue iniziale invertito | 24,22% |
| RNN senza cue | 22,66% |
| Movimento casuale legale | 21,88% |
| Memoria programmata | **100,00%** |

Contrasto RNN − FF: **+1,17 pp**, bootstrap gerarchico 95% **−16,41/+21,09 pp**, **quattro** run positive e **quattro** negative. La memoria azzerata cambia il successo di +4,30 pp (intervallo **−4,69/+13,67 pp**). Il cue invertito produce +1,95 pp (intervallo **−15,62/+18,36 pp**). **Nessuna evidenza di robusta memoria a ritardo maggiore.**

La RNN raggiunge il bivio solo nel **50,39%** degli episodi in questi mondi: la difficoltà motoria cresce con la lunghezza del corridoio e confonde la misura della memoria.

## 6. Trasferimento alla ricerca libera di cibo: ancora scarso

Gli otto campioni RNN e FF selezionati per il bivio vengono valutati **senza riaddestramento** in 12 labirinti aperti con odore del cibo, 100 mosse massime e sistema fisico M4.1.

| Agente | Cibo medio raccolto |
|---|---:|
| RNN evoluta sul bivio | **1,50** |
| Stessa RNN con odore disattivato | 1,36 |
| Feedforward evoluta sul bivio | 0,84 |
| Movimento casuale legale | **2,58** |
| Euristica basata sull'odore | **7,92** |

La differenza RNN con odore meno senza odore è **+0,14 unità di cibo**, intervallo gerarchico descrittivo **−0,32/+0,58**. **Nessuna evidenza di utilizzo utile dell'odore**, e la rete ricorrente non supera il camminatore casuale nella raccolta libera di cibo.

Questo test è solo diagnostico: l'evoluzione di M4.2 avviene sui corridoi a bivio, non su ambienti generali di foraggiamento.

## 7. Limiti di interpretazione e audit

1. **Il gate a senso unico** è un artificio controllato: garantisce che la decisione al bivio dipenda, in teoria, dall'indicazione ricordata. Non prova orientamento autonomo in un ecosistema.
2. Il contenuto del cue è fornito dall'esperimentatore. La rete non scopre da sola il significato ecologico del cibo né si assegna un obiettivo.
3. La rete feedforward ha **268 pesi contro 260** della ricorrente, non esattamente lo stesso numero; 3,1% di differenza è a vantaggio della FF.
4. **Otto genomi finali per architettura**, selezionati su mondi di validazione dopo una ricerca evolutiva; gli intervalli bootstrap gerarchici sono descrittivi, non un numero grande di repliche indipendenti.
5. La differenza in probabilità di **raggiungere il bivio** è un possibile confondente: controllare motricità del corridoio è necessario prima di affermare una vera memoria spaziale.
6. **Il senso orientato al cibo è disabilitato** nei corridoi e la rete ha input energetico costante per chiudere la fuga informativa via tempo. La rete non può usare coordinate globali o il test seed.
7. L'energia reale continua a ridursi, e un episodio che si arresta senza imboccare un ramo non viene marcato come successo. Il controllore scriptato mostra che il percorso è fisicamente risolvibile.
8. Le modifiche dei pesi avvengono **fra vite attraverso un algoritmo genetico esterno**, non durante la vita. Nessuna riproduzione intra-mondo, predatori, coscienza, codice scritto autonomamente o obiettivi scelti dal simulato.

## 8. Prossimo passo motivato dai dati (M4.3, non ancora implementato)

Per isolare davvero la memoria, serve una prova in cui **il percorso di avvicinamento non sia un collo di bottiglia**. Una possibile soluzione è separare due sottoprove:
- **Decisione pura con esperienza temporale**: il sistema riceve il cue, attraversa una sequenza di osservazioni neutre di durata variabile e arriva a un bivio dove le azioni legali sono due, mantenendo lo stesso input finale per goal opposti. Tutte le architetture devono arrivare al bivio: la misura primaria è solo la scelta corretta.
- **Navigazione autonoma con memoria**: ripristinare progressivamente movimento nel corridoio, poi ostacoli variabili, verificando che lo stesso controllo di memoria dia un vantaggio *addizionale* rispetto alla sola capacità locomotoria.

In una futura preregistrazione bisognerebbe prevedere uno stimolo ritardato insegnato con una **progressione di difficoltà fissata ex ante** (curriculum dal cue presente fino a cue che scompare), nuovi seed genetici e nuovi seed finali, reset dello stato e controllo delle azioni effettivamente sensibili al cue. Non riutilizzare i seed finali M4.2 per scegliere soglie/durate/genomi.

**Decisione:** M4.2 è ingegneristicamente completata, mostra un miglioramento di architettura, ma **NON conferma il meccanismo scientifico della memoria spaziale** secondo le condizioni poste prima dei risultati. Non trasformare la scoperta parziale in una rivendicazione di comportamento intelligente emergente.

## File e riproducibilità

L'artefatto della campagna [GitHub Actions #44908425038](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44908425038) contiene:
- `summary.json` con tutte le medie, intervalli gerarchici e gate;
- `genotypes.json` con 16 reti selezionate, parametri completi e generazioni di selezione;
- `train_validation.csv` con 96 righe (8 evoluzioni ×2 architetture ×6 generazioni);
- `delayed_trial_metrics.csv` con **5.376 episodi test** (8 seed genetici ×48 coppie di mondi ×2 obiettivi ×7 bracci);
- `reactive_transfer_metrics.csv` con 480 prove di trasferimento (8×12 mondi×5 bracci);
- `example_main.svg` e `example_shifted.svg` con geometria, cibo e traiettoria registrata; rispettive tracce JSON.

La tabella persistente [M4_2_success_by_genetic_run_2026-11-21.csv](M4_2_success_by_genetic_run_2026-11-21.csv) registra 8 run genetiche ×2 famiglie di mondi ×7 controlli; le metriche complete sono disponibili nell'artefatto originale fino alla scadenza prevista e sono riproducibili dal codice e dai seed congelati.
