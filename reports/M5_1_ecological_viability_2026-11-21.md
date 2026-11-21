# M5.1 — La vita è fisicamente sostenibile, ma non emerge dai genomi neurali casuali

**Campagna: 21 novembre 2026.** [Protocollo M5.1 registrato prima dell'implementazione](../docs/M5_1_PROTOCOL.md). [Codice](../neuroevolution/m5_1.py), [test](../tests/test_m5_1.py), [PR #29](https://github.com/kkongiu/neuroevolution-lab/pull/29), [fix esclusivamente grafico PR #30](https://github.com/kkongiu/neuroevolution-lab/pull/30). **Workflow finale concluso con successo:** [GitHub Actions #44913351669](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44913351669). **Artefatto:** `m5-1-ecological-viability` id **14710911768**, contenente `summary.json`, **104 righe** di `world_metrics.csv`, **936 checkpoint** di `population_checkpoints.csv`, SVG popolazioni e mappe, eventi e genealogie di mondi rappresentativi; scadenza dell'artefatto prevista 21 dicembre 2026.

### Stato degli esiti preregistrati

- **PASSA — fattibilità fisica**: la strategia **programmata** che segue odore e ricorda il numero di visite precedenti mantiene organismi vivi **in 16/16 mondi principali** al tick 400 e raggiunge **generazione ≥3 in 16/16**. Il protocollo chiedeva ≥8/16 per entrambe le condizioni.
- **PASSA — vantaggio del foraggiamento**: anche il controller locale *greedy odor* senza memoria delle visite raccoglie più cibo del controller con movimento casuale in **16/16 mondi**: differenza media **+179,375 unità/mondo**, bootstrap descrittivo appaiato al 95% **[+130,125, +228,875]**. Richiesti ≥12/16 mondi favorevoli e lower bound >0.
- **FALLISCE — rete neurale originale**: la popolazione M5.0 con genomi ereditati e mutati si estingue in **16/16 mondi**, nonostante le stesse risorse e leggi fisiche. I due controllori programmati greedy sopravvivono entrambi in **16/16**; il camminatore casuale con riproduzione programmata in **10/16**.
- **OOD più difficile**: su otto mondi nuovi, con più muri, odore più corto e meno rigenerazione, rete originale **0/8**, greedy locale **7/8**, greedy + visite **8/8**, random legale **1/8** superstiti a 400 tick.
- **Controlli**: su quattro mondi senza riproduzione, la popolazione programmata è completamente estinta a fine orizzonte, **zero nascite**; su quattro mondi senza rigenerazione di cibo, sono nati figli ma tutti si estinguono, **soltanto le 20 risorse iniziali**. Il libro mastro energetico chiude a zero in tutti i 936 checkpoint e tutti i test di regressione sono verdi.

## 1. Disegno

Nessuna modifica al motore fisico [M5.0](../neuroevolution/m5_0.py): 15×15 con ostacoli, 20 fondatori con identici genomi casuali e posizioni tra i bracci dello stesso seed, cibo rinnovabile, 400 tick, metabolismo 1, cibo +18, età massima 100 e nascita che costa al genitore 20 energia (12 al nuovo corpo, 8 dissipati), più il costo metabolico del turno.

Quattro politiche fanno esperienze diverse **nello stesso tipo di mondo**:
1. `neural_heritable`: controller RNN M5.0 non alterato; i genomi influenzano le decisioni, inclusa la scelta di riprodursi, e vengono ereditati/mutati nelle nascite.
2. `greedy_odor`: policy PROGRAMMATA (non evoluta), sceglie una direzione locale con cibo adiacente o odore maggiore, altrimenti movimento casuale legale. Riproduce con regola esplicita energia≥46, età≥4 e capacità fisica.
3. `greedy_novelty`: policy PROGRAMMATA, valuta cibo locale, odore e piccola preferenza per celle visitate meno spesso dallo stesso individuo. La memoria delle visite è fornita dal programma; anche la riproduzione usa una soglia programmata.
4. `random_legal`: movimento casuale legale con la **stessa regola programmata di riproduzione** delle altre due policy scriptate; non legge odore né cibo per scegliere direzione.

**IMPORTANTE:** le tre strategie scriptate **NON** fanno evolvere capacità motorie attraverso il genoma; anche quando nascono figli con genomi numerici, questi pesi NON determinano le loro azioni. La conservazione della genealogia è un esperimento di riproduzione fisica programmata, non un successo evolutivo neurale.

**Confronto non perfettamente simmetrico nelle azioni:** la rete originale sceglie autonomamente quando riprodursi; le policy scriptate usano una soglia energetica esplicita, un indicatore di età non esposto alla RNN e, per novelty, una memoria delle visite che la RNN non possiede. Pertanto il divario *non* può essere attribuito **solo** all'orientamento verso il cibo. Potrebbero contribuire anche politica riproduttiva, controllo motorio, percezione e memoria locale. I **costi fisici della nascita**, comunque, sono identici.

Nuovi world seed mai impiegati in M5.0: **73000–73015 main** (16 mondi) e **74000–74007 shifted** (8). Nessuna scelta post-hoc di genomi, threshold o mondi. Nessun aggiustamento al variare dei risultati. Gli organismi non hanno accesso al codice, a file, rete o tool.

## 2. Esiti principali dopo 400 tick

| Politica | Mondi con ≥1 vivo /16 | Organismi vivi medi | Nascite medie | Cibo mangiato medio | Generazione massima media | Cibo rimasto medio /32 |
|---|---:|---:|---:|---:|---:|---:|
| **RNN originale con ereditarietà** | **0** | **0,00** | **21,38** | **54,94** | **2,94** | **32,00** |
| **Greedy odore programmato** | **16** | **13,56** | **112,50** | **404,19** | **8,75** | **4,31** |
| **Greedy + memoria visite programmata** | **16** | **13,94** | **112,63** | **408,13** | **8,94** | **3,81** |
| Casuale legale + riproduzione programmata | 10 | 5,25 | 63,13 | 224,81 | 7,00 | 29,31 |

**Ambiente capace di sostenere un controllo funzionante** con le stesse regole di energia e risorse: la rete neurale rimane a zero, i controller programmati seguono riproduzioni ripetute fino al tick 400 in tutti i mondi normali.

Una seconda osservazione importante: il semplice greedy odore è quasi pari al greedy+visite. L'aggiunta della memoria individuale delle celle porta **+3,94 cibi**, intervallo bootstrap descrittivo **[+0,50,+7,44]**: effetto modesto rispetto al salto fra network casuale e controller diretto. Non è necessario simulare una mappa completa per sopravvivere in questi mondi.

## 3. Popolazione e risorse nel tempo

Medie su tutti i 16 mondi main:

| Tick | RNN originale: vivi | Greedy odore: vivi | Greedy+visite: vivi | Random legale: vivi | Cibo rimasto RNN /32 |
|---:|---:|---:|---:|---:|---:|
| 0 | 20,00 | 20,00 | 20,00 | 20,00 | 20,00 |
| 50 | 11,44 | 16,94 | 17,25 | 17,56 | 18,88 |
| 100 | **1,31** | 9,06 | 10,12 | 5,88 | **31,94** |
| 150 | 0,56 | 14,31 | 14,62 | 6,94 | 32,00 |
| 200 | 0,38 | 12,81 | 14,44 | 7,81 | 32,00 |
| 300 | 0,00 | 13,81 | 12,50 | 6,62 | 32,00 |
| 400 | **0,00** | **13,56** | **13,94** | 5,25 | **32,00** |

È **l'accesso funzionale al cibo**, non l'assenza globale di cibo, il collo di bottiglia immediatamente osservabile; quando la rete si estingue, il cibo è pieno. Questa non è però una prova causale che il *solo* foraggiamento spieghi la scomparsa (le policy hanno anche regole riproduttive differenti).

## 4. Prestazione del comportamento

| Metrica cumulata o media per mondo | RNN ereditaria | Greedy odore | Greedy+visite | Random legale |
|---|---:|---:|---:|---:|
| Cibo per 100 azioni motorie | **3,83** | **7,00** | 6,96 | 5,79 |
| Scelte motorie allineate con odore più forte, fra le misurabili | **38,47%** | **100%** | 84,20% | 39,90% |
| Correzioni di azioni impossibili /mondo | **773,25** | 0 | 0 | 0 |
| Desideri neurali di riprodursi in stato impossibile /mondo | **282,94** | 0 | 0 | 0 |

Questi conteggi non sono eventi indipendenti né il numero di tentativi per un singolo individuo: sono aggregati a livello di mondo e confondono parzialmente durata complessiva della popolazione ed efficienza dei singoli animali. Il punto è la bassa correlazione fra politica neurale non addestrata e segnale alimentare, insieme alla tendenza a preferire azioni bloccate.

**Nota di audit:** le statistiche sull'allineamento dell'odore sono condizionate a *mosse effettivamente tracciate* e presenza di odore misurabile, non sono sempre confronti con identici contesti sensoriali. Le azioni morte/termine possono essere contate nel totale delle azioni motorie; la formula del tasso usa il contatore delle mosse del motore originale.

## 5. Mondi shifted più difficili

Otto mondi con ostacoli più densi, odore raggio 2 e rigenerazione rallentata:

| Strategia | Mondi sopravvissuti /8 | Vivi medi | Nascite medie | Cibo medio |
|---|---:|---:|---:|---:|
| Neural original | **0** | **0,00** | **19,63** | **50,75** |
| Greedy odore | **7** | 8,88 | 85,50 | 276,13 |
| Greedy + visite | **8** | 10,25 | 88,25 | 292,50 |
| Random legale | 1 | 0,50 | 24,38 | 73,50 |

La sostenibilità delle politiche programmate resiste a un ambiente più ostile, ma ancora una volta **non è un successo evolutivo spontaneo**.

## 6. Controlli, verifiche e audit dell'esecuzione

- Suite regressiva con tutti gli esperimenti storici; test delle policy locali, nessuna collisione, energia chiusa, nascita valida, stessa rigenerazione potenziale e stessi genomi fondatori.
- Test di **equivalenza causale del braccio neurale**: sui medesimi seed produce esattamente le statistiche fisiche del modulo M5.0 originale; M5.0 non è stato ritoccato.
- Controlli senza nascite: quattro mondi di sola alimentazione e locomozione, sempre 0 nati, estinti alla scadenza biologica dei fondatori. Controlli senza ricrescita del cibo: quattro mondi, 20 pellet iniziali in totale, nessun nuovo alimento; nessun superstite a tick 400.
- Ogni checkpoint (936) ha residuo del bilancio energetico **0,0** entro la precisione numerica. Non esistono fonti energetiche esogene non contabilizzate.
- La prima esecuzione [#44913028853](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44913028853) calcolò **tutti i mondi e le summary**, poi fallì **solo** alla serializzazione SVG di una popolazione neurale estinta per una comprehension che produceva un dizionario vuoto al posto di una lista di corpi. Il [PR #30](https://github.com/kkongiu/neuroevolution-lab/pull/30) ha corretto soltanto la conversione dei dati per l'immagine e aggiunto un test di popolazione nulla; **nessun cambiamento a ecologia, policy, seed, soglie o metriche**. Il replay identico [#44913351669](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44913351669) è terminato con successo e ha preservato tutti i file.
- **L'unità indipendente è il mondo.** Ogni braccio è appaiato sugli stessi world seed, ma la realizzazione dei tentativi di crescita alimentare può divergere quando cambiano occupazione ed eventi. Nessun test considera ogni individuo nato o ogni tick come una replica indipendente.

## 7. Conseguenza scientifica

**Non possiamo più attribuire l'estinzione generalizzata di M5.0 a un ecosistema impossibile da abitare:** un controller locale semplice e programmato raggiunge una popolazione non estinta dopo 400 tick in tutti i 16 mondi principali e arriva in media alla nona generazione. Per contro, **l'evoluzione numerica spontanea nella configurazione M5.0 non produce comportamenti abbastanza coordinati da permanere**, nonostante la riproduzione in-world.

**Non abbiamo dimostrato che l'evoluzione spontanea sia impossibile.** Abbiamo dimostrato che, con 401 pesi neurali inizialmente casuali, mutazione gaussiana di ogni peso per nascita, un solo mondo finito per replica e la selezione attraverso nascita/morte, la competenza ecologica necessaria non è emersa prima dell'estinzione.

### Nuovo esperimento consigliato, non ancora implementato

**M5.2 dovrebbe separare in un disegno fattoriale le due competenze:**
- **Movimento:** RNN originale, oppure stessa RNN con un *aiuto motorio locale* che la orienta stocasticamente verso l'odore, con intensità preregistrata; in una prova totalmente distinta, un controller molto più piccolo (**pochi parametri** invece di 401) che legge gli stessi sensori e sceglie autonomamente senza policy odore codificata.
- **Riproduzione:** scelta neurale originale oppure la stessa soglia energetica programmata usata per tutti i baseline. Le **quattro combinazioni** permettono di capire se sia la ricerca del cibo, la decisione riproduttiva o la loro interazione a causare l'estinzione.
- Stessa energia, spazio, cibo, risorse e mondi per tutti; nuovi seed, protocollo congelato, stesso budget di popolazione, controlli su genomi ereditati/clonati/random e un vero gate di **persistenza con vantaggio ereditario**.
- Solo quando un sistema neurale riproducibile riesce a persistere, testare se le **mutazioni creano una capacità nuova** rispetto alla selezione di genomi fondatori già efficaci.
- Non risolvere il problema incollando semplicemente il programma greedy dentro la rete e chiamarlo emergenza.

**Decisione:** M5.1 **PASSA la fattibilità** e il controllo di foraggiamento programmato. L'esito opposto del sistema neurale rivela che il limite scientifico attuale è l'assenza di un ponte evolutivo praticabile fra controller casuale e organismo sostenibile, non la pura scarsità del mondo. Persistenza dei controlli scriptati **non è** vita emersa spontaneamente.
