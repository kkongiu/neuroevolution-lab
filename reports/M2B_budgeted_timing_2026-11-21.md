# M2B — Rapporto empirico: rinnovo neurale a dose fissa (21 novembre 2026)

**Risultato primario: NON CONFERMATO.** Il controllore che osserva lo stato reale della rete non supera quello che osserva esclusivamente il numero del compito, a parità esatta di unità rinnovate.

- [Protocollo congelato prima del test](../docs/M2B_PROTOCOL.md)
- [Codice](../neuroevolution/meta_budget.py)
- [GitHub Actions #44883457787](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44883457787) — completata, tests OK
- [Pull request #5](https://github.com/kkongiu/neuroevolution-lab/pull/5) — integrata in main
- Artefatto Actions `m2b-budgeted-timing`, ID **14698599114**, prevista scadenza **21 dicembre 2026**.

## Perché questo test era necessario

[M2](M2_adaptive_2026-11-21.md) aveva scelto di non rinnovare MAI: la funzione obiettivo penalizzava gli interventi e la dimenticanza. Con la stessa dose nulla tutte le strategie diventavano identiche, rendendo impossibile misurare il tempismo.

In M2B la dose è obbligatoria ed **esattamente uguale**, evitando questa degenerazione:

- 8 compiti successivi e 7 transizioni.
- 16 unità ReLU, massimo 3 unità rinnovabili per transizione.
- **10 unità rinnovate complessivamente per ogni sequenza**, per tutte le cinque strategie appaiate.
- Ricerca evolutiva dei controller (8 candidati × 5 generazioni per modalità): meta-training 4000–4003, meta-validation 5000–5002; 16 test isolati 6000–6015.
- I controller sono due e si evolvono indipendentemente: uno vede errori di training, attività delle unità e tempo; l'altro vede **solo il tempo**.
- 64 SGD update per task, stessa rete iniziale, stessi esempi, stessi task e holdout per seed.
- Obiettivo della selezione: AUC di apprendimento meno 0,05 × dimenticanza; il costo in numero di rinnovi è identico, quindi il suo termine è costante.

## Risultati finali

**AUC = area normalizzata sotto la curva di accuratezza durante l'apprendimento, non ROC AUC.**

| Strategia | AUC media (16 seed) | Accuratezza finale | Dimenticanza media | Unità rinnovate / sequenza |
|---|---:|---:|---:|---:|
| **Adaptive (stato + tempo)** | **65,50%** | 75,74% | 22,24 pp | **10** |
| **Time-only (solo tempo)** | **65,75%** | 75,74% | 22,42 pp | **10** |
| Uniforme | **65,47%** | 75,58% | 22,21 pp | **10** |
| Tempi permutati a dose uguale | **65,39%** | 75,37% | 22,18 pp | **10** |
| Selezione casuale delle unità, stesse dosi e tempi dell'adaptive | **64,25%** | 74,44% | 22,53 pp | **10** |
| Nessun rinnovo (fuori budget) | **64,16%** | 74,11% | 21,11 pp | 0 |
| Rinnovo massimo fisso (fuori budget) | **65,99%** | 76,45% | 23,49 pp | 21 |
| Nuova rete a ogni task (controllo diagnostico) | **67,94%** | 76,23% | 31,26 pp | nuova rete |

### Confronti appaiati, metrica primaria e secondarie

| Adaptive meno... | Differenza di AUC | Bootstrap descrittivo 95% | Segni su 16 seed |
|---|---:|---:|---|
| **Time-only (PRIMARIA)** | **−0,25 pp** | **−0,75 / +0,20 pp** | 7 positivi / 9 negativi |
| Uniforme | **+0,03 pp** | −0,61 / +0,70 pp | 8 / 8 |
| Tempi permutati | **+0,11 pp** | −0,43 / +0,70 pp | 8 / 8 |
| Selezione casuale delle unità | **+1,25 pp** | +0,32 / +2,42 pp | 9 / 7 |
| Nessun rinnovo | **+1,34 pp** | +0,80 / +1,93 pp | 14 / 2 |
| Rinnovo massimo fisso | **−0,50 pp** | −1,01 / −0,01 pp | 4 / 12 |
| Nuova rete | **−2,44 pp** | −4,80 / −0,19 pp | 5 / 11 |

Gli intervalli si ottengono con un bootstrap di differenze appaiate **a livello dei 16 seed**, non per singolo task o singolo esempio. Sono descrittivi; l'esperimento testa più confronti e non fornisce evidenza universale.

**Il criterio primario preregistrato fallisce.** Un controller che utilizza segnali di training non produce un vantaggio positivo sul controller che vede esclusivamente il tempo. La differenza di −0,25 pp è compatibile con zero.

### C'è stata una modifica comportamentale autentica?

Il controller adaptive selezionato ha coefficienti non nulli per segnali della rete; l'ablation che nasconde tali segnali cambia la richiesta in **23 delle 112 transizioni** (16 × 7). Il controller ha quindi sviluppato **una risposta operativa dipendente dallo stato**, ma non siamo autorizzati a definirla vantaggiosa.

Istogramma delle **richieste**, prima dell'applicazione della quota: 57 richieste di 0 neuroni e 55 richieste di 2 neuroni.

Istogramma degli **interventi effettivi**: 32 interventi da 0 neuroni; 9 da 1; 62 da 2; 9 da 3. Il vincolo del budget ha **forzato 25 interventi** rispetto alla richiesta originaria. La variabilità indotta dal vincolo **non** conta come scelta autonoma.

Il controller time-only ha un unico programma di rinnovo per tutti i seed, per esempio:

`[2, 2, 2, 2, 2, 0, 0]`

Un esempio di sequenza adattiva osservata sul seed 6000 è:

`[2, 2, 2, 0, 0, 1, 3]`

Quindi abbiamo una variazione della decisione in funzione dell'esperienza, ma non la prova che questa variazione migliori l'apprendimento.

## Cosa sembra funzionare?

Il confronto adattivo versus selezione *casuale delle unità* a pari dose/timing è in media favorevole: +1,25 pp. Questo è coerente con i piloti precedenti M1-P/M1-P2: scegliere unità poco attive può essere utile. Va però interpretato cautamente (9 seed positivi e 7 negativi, intervallo descrittivo e più confronti).

Il vantaggio rispetto a nessun rinnovo (+1,34 pp) dimostra solo che **un certo rinnovo strutturale** può aiutare in questo ambiente, non che le scelte dipendenti dallo stato siano superiori a una programmazione semplice.

Il rinnovo massimo fisso porta la AUC a 65,99%, con un budget oltre il doppio (21 contro 10). Non è un confronto a dose equivalente.

## Conclusione e decisione metodologica

**M2B non conferma che imparare a scegliere il momento del rinnovo migliori la prestazione rispetto a una regola temporale fissa.** Esiste una sensibilità allo stato nei comandi del controller, ma non un beneficio generalizzabile su questi 16 seed.

Non cambieremo a posteriori il coefficiente del controller, il budget o le soglie dei segnali per trasformare questo esperimento in un successo. L'ipotesi primitiva — reti che apprendono a ri-scrivere le regole del proprio apprendimento — rimane lontana: qui un algoritmo esterno evolve cinque coefficienti di un semplice controllore.

Il prossimo test scientificamente interessante non è moltiplicare a caso le mutazioni. Conviene costruire un **ambiente in cui lo stato della rete abbia un ruolo verificabile**: task che si ripresentano con probabilità variabile, costi espliciti di oblio, novità segnalata o inattesa, budget computazionali e temporali equivalenti. Pre-registrare ablation *state-aware versus time-only* con nuovi task e seed; includere overhead CPU. Se in quell'ambiente non emerge un vantaggio, trattarlo come nuova evidenza negativa.

## Provenienza dei dati

L'esecuzione GitHub Actions conserva:
- `summary.json`: pesi delle politiche, cronologia meta-evolutiva, aggregati e intervalli;
- `meta_training.jsonl`: generazioni dei due controller;
- `task_metrics.csv`: 16 × 8 × 8 = **1.024** misurazioni individuali;
- `test_seed_metrics.csv`: **128** righe seed×strategia, azioni richieste e realmente eseguite.

Per preservare i valori principali dopo la scadenza dell'artefatto, vedere [M2B_test_seed_auc_2026-11-21.csv](M2B_test_seed_auc_2026-11-21.csv).
