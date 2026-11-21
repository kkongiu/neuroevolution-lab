# M4.1 — Navigazione legalizzata, evoluzioni indipendenti e ricerca del cibo

**Data:** 21 novembre 2026. **Esito primario preregistrato: NON CONFERMATO.**  
**Effetto della maschera motoria: POSITIVO e replicato, ma programmato dal ricercatore.**  
**Superiorità evolutiva rispetto a una strategia di movimento casuale già legalizzata: NON DIMOSTRATA.**

- [Protocollo M4.1 preregistrato prima del codice](../docs/M4_1_PROTOCOL.md).
- [Codice evolutore e simulatore](../neuroevolution/m4_1.py), [test](../tests/test_m4_1.py), [PR #21](https://github.com/kkongiu/neuroevolution-lab/pull/21) integrata in main.
- [Esecuzione GitHub Actions #44906089697](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44906089697): test e campagna completati con successo.
- [CSV permanente con gli 8 seed genetici × 2 tipi di mondi](M4_1_food_by_genetic_run_2026-11-21.csv). Le medie nel CSV sono arrotondate a 4 decimali.
- **Artefatto originale** GitHub Actions: `m4-1-masked-navigation`, ID **14707414900**, scadenza prevista **21 dicembre 2026**. Include *summary.json*, *champions.json* (16 genomi congelati), *train_validation.csv*, *test_world_metrics.csv* (2.304 righe), *null_food.csv* (64 osservazioni), 2 mappe SVG e 2 registri di traiettoria JSON.

## 1. Che cosa abbiamo davvero cambiato rispetto a M4.0

[M4.0](M4_0_survival_world_2026-11-21.md) ha costruito il primo organismo digitale simulato, ma l'organismo evoluto raccoglieva **2,19 cibi** contro **3,69** di un camminatore casuale che evita i muri. Sbattendo contro le pareti mediamente **17,66 volte per vita**, consumava energia senza muoversi.

M4.1 mantiene **la stessa fisica**, la disponibilità di energia/cibo, il sistema di sensori (incluso l'odore progettato) e la fitness. Cambia **solo la legalità motoria e il disegno evolutivo**:

1. Un decoder aggiunto dopo il calcolo della rete sceglie l'azione legale col logit più alto se l'azione preferita è bloccata. Il decoder usa solo i 4 sensori di parete immediata, non mappe, posizioni del cibo o percorsi ideali.
2. Il decoder può essere acceso o spento **sullo stesso identico genoma**, mantenendo fissi pesi, sensori e mondo: ciò isola l'effetto dell'assistenza motoria.
3. **8 inizializzazioni genetiche indipendenti**, ognuna evolve due popolazioni inizialmente identiche: una con maschera durante il training e una senza.
4. Per ogni coppia di genomi, valutazione incrociata: *training mascherato/non mascherato × esecuzione mascherata/non mascherata*.
5. Mondi ruotati casualmente e posizione iniziale scelta fra celle raggiungibili con almeno due movimenti validi, per evitare la ripetizione del centro e della stessa orientazione.
6. Ogni generazione affronta **6 nuovi mondi**; 4 mondi di validazione indipendenti selezionano un campione **solo in base alla validazione**. Nessun test finale è usato nella selezione.

Il sistema **non impara i pesi durante una vita** e non si riproduce nel mondo. È selezione evolutiva esterna di reti ricorrenti a topologia fissa.

## 2. Campagna preregistrata, selezione senza test leakage

- **8 genetic seed:** 81000–81007.
- **2 addestramenti indipendenti per genetic seed**: maschera attiva/disattiva.
- **6 generazioni × 10 genomi × 6 mondi di training per generazione = 360 prove di training per evoluzione**, quindi **5.760** candidate-world training episodes sull'intera campagna.
- **24 valutazioni di validazione per evoluzione**, quindi 384 episodi di validazione complessivi.
- Mondi di training **45000–45035**; validazione **46000–46003**.
- Nuovi mondi finali **47000–47023** (24), stress test **48000–48011** (12), completamente separati da M4.0.
- Su ogni mondo finale sono misurate **8 condizioni**: 2×2 decodifica/rete, una rete casuale per genetic seed con decoder, movimento casuale lecito, *greedy odor*, e maschera evoluta con odore disabilitato.

Gli 8 genomi evoluti con maschera **non sono stati scelti retroattivamente per il miglior rendimento finale**: tutti sono conservati e inclusi nelle statistiche. Le **192 combinazioni di 8 genomi × 24 mondi** non equivalgono a 192 repliche indipendenti di evoluzione. Gli intervalli riportati ricampionano gerarchicamente **interi genetic seed** e **interi world seed**, mantenendo gli agenti appaiati sullo stesso mondo.

## 3. Test principale — 24 mondi, 8 genomi evoluti

| Agente/condizione | Cibo medio raccolto (su 12) | Passi/vita | Urti effettivi/vita | Scelte illegali corrette/vita |
|---|---:|---:|---:|---:|
| **Rete evoluta addestrata con maschera, eseguita con maschera** | **2,245** | **51,54** | **0** | **23,73** |
| Stesso genoma evoluto, eseguito senza maschera | 1,448 | 37,45 | 19,99 | 0 |
| Rete evoluta senza maschera nel training, eseguita senza maschera | 1,609 | 41,25 | 15,78 | 0 |
| Rete evoluta senza maschera nel training, eseguita con maschera | 2,266 | 52,86 | 0 | 14,66 |
| **Rete neurale casuale, con maschera** | 0,901 | 36,57 | 0 | 16,17 |
| **Camminatore casuale che evita i muri** | **2,292** | 55,12 | 0 | 0 |
| **Euristica che segue l'odore, evitando i muri** | **8,875** | 76,46 | 0 | 0 |
| Rete evoluta con maschera, ma odore disattivato | 2,313 | 51,30 | 0 | 22,00 |

### Confronti appaiati — 95% bootstrap gerarchico descrittivo

| Test | Δ cibo raccolto | Intervallo 95% | Genetic run favorevoli |
|---|---:|---:|---:|
| **Evoluta mascherata − casuale senza urti (PRIMARIO)** | **−0,047** | **[−1,156, +1,057]** | **4 positivi, 3 negativi, 1 pari su 8** |
| Evoluta mascherata − neurale casuale mascherata (necessario) | **+1,344** | **[+0,677, +2,042]** | **8/8 positivi** |
| Evoluta mascherata − greedy odor | **−6,630** | **[−8,182, −5,010]** | **0/8 positivi** |
| **Stesso genoma, maschera ON − OFF** (addestrato con maschera) | **+0,797** | **[+0,323, +1,266]** | **8/8 positivi** |
| Stesso genoma addestrato senza maschera, esecuzione con − senza maschera | **+0,656** | **[+0,260, +1,083]** | **8/8 positivi** |
| **Addestrato con maschera − senza maschera, entrambi eseguiti con maschera** | **−0,021** | **[−0,630, +0,698]** | 4/8 positivi |
| Rete evoluta con odore − stessa rete senza odore | **−0,068** | **[−0,531, +0,385]** | 2/8 positivi |

**La soglia primaria definita prima del test non è soddisfatta**: la differenza di cibo rispetto al camminatore lecito è leggermente negativa, l'intervallo include zero e i genetic seed favorevoli sono solo **4/8**, sotto il requisito **≥6/8**.

L'effetto della maschera **è un successo causale ingegneristico**: sulla *stessa rete*, impedire i movimenti illegali aggiunge in media **0,80** cibi, riduce di circa **20** gli urti reali per vita e produce un incremento positivo in tutti gli otto genetic seed. Questo vantaggio è **prodotto dal decoder programmato da noi**, non dimostra che la rete abbia imparato ad evitare i muri.

Le reti selezionate evolutivamente battono in modo abbastanza sistematico una rete **casuale mascherata**; tuttavia non superano il movimento casuale ben formato. La fitness stessa non garantisce orientamento alimentare complesso.

**Il train con maschera non dà vantaggio dimostrato** se confrontiamo i due genomi con la maschera ON in esecuzione: 2,245 vs 2,266 cibi. La maggior parte del vantaggio deriva dall'effetto immediato del decoder, non dall'apprendimento evolutivo specifico del suo uso.

**Nessuna dipendenza dall'odore dimostrata**: disabilitarlo non peggiora in media la raccolta alimentare della rete mascherata. Il controllo greedy che usa lo stesso odore raccoglie **8,875 cibi**, dimostrando che il segnale ambientale è informativo, ma la rete non sembra sfruttarlo efficacemente.

L'organismo mascherato mangia almeno una risorsa nell'**80,21%** dei confronti genoma–mondo principali; questo da solo non soddisfa il criterio primario.

## 4. Generalizzazione: stress test su 12 mondi più ostili

| Agente | Cibo medio (su 8) | Passi/vita | Urti effettivi/vita |
|---|---:|---:|---:|
| Evoluta con maschera | **1,063** | 38,95 | 0 |
| Evoluta senza maschera nel deployment | 0,594 | 28,41 | 16,66 |
| Evoluta senza maschera nel training, con maschera nell'esecuzione | 0,792 | 36,09 | 0 |
| Rete neurale casuale con maschera | 0,510 | 31,60 | 0 |
| Casuale che evita i muri | **1,083** | 40,33 | 0 |
| Greedy odor | **2,667** | 55,83 | 0 |
| Evoluta con maschera e senza odore | 0,958 | 37,40 | 0 |

- **Evoluta − camminatore casuale**: −0,021 cibi; bootstrap 95% **[−0,688, +0,635]**, 2/8 run favorevoli, 5/8 sfavorevoli e una pari.
- **Evoluta − rete casuale mascherata**: +0,552 cibi, **[+0,188, +0,948]**.
- **Stesso genoma, decoder ON − OFF**: +0,469 cibi, **[+0,146, +0,917]**.
- **Evoluta − greedy**: −1,604 cibi, **[−2,781, −0,490]**.

Lo stress test conferma che la maschera è utile anche su mondi differenti, ma **non dimostra generalizzazione di una migliore strategia alimentare rispetto a una ricerca casuale lecita**. Il **59,38%** dei confronti genoma–mondo raccoglie almeno un cibo, meno del test principale.

## 5. Controlli negativi e prove di qualità

- Su **64 prove di mondo senza cibo** (8 genomi × 8 mondi), il sistema non crea artificialmente né cibo né energia. Risultato: **0 cibi in tutte le prove**; durata media 24 passi, coerente con l'energia iniziale senza recupero; preferenze motorie illegali corrette: **8,59 per vita**.
- Tutte le unit test passano, compresi i test storici fino a M4.0.
- Motore 2D riproducibile, food spawning in celle raggiungibili, rotazioni, start non occupati dal cibo, 2+ uscite legali, semantica della maschera testata direttamente su logit noti, confronto dello stesso genoma con/ senza maschera, confronto delle evoluzioni a budget identico.
- I campioni genetici sono scelti **solo sulle 4 mappe di validazione**. I migliori campioni provengono da generazioni diverse; questo evita di assumere che l'ultima generazione sia sempre migliore.
- Il metodo di inferenza è descrittivo e può essere sensibile a numero ridotto di **8 repliche evolutive** e **24 mondi condivisi**. Non sono esperimenti con ecosistemi reali.

## 6. Interpretazione e decisione

**M4.1 risolve una debolezza motoria, non l'incapacità di cercare cibo.** L'assistenza motoria riduce a zero gli urti ma non insegna a usare l'odore. I neuroni continuano a proporre in media **23,73 azioni impossibili per vita**, e il decoder corregge quei movimenti. La rete raggiunge il livello della ricerca casuale lecita, non quello di una politica alimentare efficace.

L'esperimento impedisce di attribuire erroneamente alla rete ciò che è stato programmato nel decoder. Il fatto che il training con maschera non sia superiore al training senza maschera, a decoder finale uguale, rafforza la distinzione: **il beneficio dimostrato è fisico/ingegneristico, non una prova di intelligenza emergente**.

È comunque un progresso sperimentale: una popolazione evolutiva replicata, mondi con posizione iniziale variabile, un problema reale di risorse/energia, e controlli che permettono di falsificare affermazioni troppo forti.

### Proposta per M4.2

Prima di aggiungere predatori o riproduzione dentro il mondo, il problema da isolare è **l'uso causale dei sensori e l'esplorazione efficace**:

- Separare task *reactive* (il cibo è visibile/odorabile nelle vicinanze) da task *memory-required* (seguire un indizio temporaneo, poi attraversare zone in cui non è osservabile). In entrambi mantenere decoder legale.
- Confrontare la rete ricorrente con una politica **senza memoria ricorrente ma con identico budget di parametri/evoluzione**, oltre a random legale e greedy. Una ricorrenza presente non prova memoria effettivamente usata.
- Preregistrare un'analisi di sensibilità odore **con e senza** forte segnale locale e magari un nuovo fitness con componenti di esplorazione, senza far trapelare la posizione vera del cibo nell'input.
- Usare **nuovi** seed genetici e dei mondi, una campagna confermativa e un confronto stratificato per difficoltà. Non adattare M4.0 o M4.1 ai test già letti.

Reproduzione in-world e predatori sono ancora obiettivi successivi, non funzionalità già presenti.

### Provenienza

[GitHub Actions #44906089697](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44906089697) contiene:
- `summary.json`: aggregati e intervalli bootstrap gerarchici, per-run e per-world delte, criteri di conferma.
- `champions.json`: **16 genomi** congelati (8 con decoder, 8 senza) con seed di ricerca e generazione selezionata.
- `train_validation.csv`: **96 righe** di statistiche generazione × replica × trattamento, con training e validation separati.
- `test_world_metrics.csv`: **2.304 risultati agent-world-run** (8 genetic seed × 36 mondi × 8 strategie).
- `null_food.csv`: 64 controlli di mondi privi di cibo.
- `example_main.svg` / `example_shifted.svg` e JSON con le traiettorie passo-passo.

Il rapporto e il [CSV compatto dei 16 gruppi run/test](M4_1_food_by_genetic_run_2026-11-21.csv) sono conservati stabilmente nel repository. L'intero esperimento è deterministico e ricostruibile dal codice e dai seed del protocollo anche dopo la scadenza dell'artefatto GitHub.
