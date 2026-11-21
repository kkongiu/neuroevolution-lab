# M5.0 — Evoluzione dentro l'ecosistema: nascite e generazioni reali, ma nessuna popolazione persiste

**Campagna:** 21 novembre 2026. **Protocollo congelato PRIMA del codice e dei test finali:** [docs/M5_0_PROTOCOL.md](../docs/M5_0_PROTOCOL.md).

**Esito:** simulatore ecologico verificato, numerose nascite e discendenze geneticamente tracciate, ma **ipotesi primaria di vantaggio dell'ereditarietà con mutazione NON confermata**; **tutte le popolazioni si estinguono**, anche in presenza di cibo rigenerato. Nessuna prova di selezione adattativa cumulativa robusta o di intelligenza ecologica emergente.

- [Codice](../neuroevolution/m5_0.py), [test](../tests/test_m5_0.py), [PR #27 integrata](https://github.com/kkongiu/neuroevolution-lab/pull/27).
- [GitHub Actions #44911495711](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44911495711): workflow concluso con **successo**, test di regressione e simulazioni preregistrate complete.
- [Archivio permanente CSV dei 60 mondi/bracci](M5_0_world_metrics_2026-11-21.csv). Tutte le sei medie nascite e cibo raccolto per split/braccio sono state riconciliate con l'artefatto originale.
- **Artefatto originale** `m5-0-continuous-ecology` ID **14709528861**, scadenza prevista **21 dicembre 2026**. Comprende `summary.json`, `seed_metrics.csv` (64 righe: 60 principali+shifted, 4 controlli senza cibo), `population_checkpoints.csv`, log completi di parentela e eventi per il primo mondo di ciascuna split/braccio, e due mappe SVG.

## 1. Perché M5.0 è diversa da M4.3

In M4 la ricerca era guidata da un ricercatore esterno: creare un esercizio, scegliere i migliori cervelli sulla fitness, propagare i pesi per generazioni di reti e valutare i campioni. In M5.0 **non esiste alcuna fitness esterna, scelta dei migliori genomi o generazione forzata**.

Venti organismi neurali occupano contemporaneamente un mondo **15×15** con ostacoli, cibo iniziale, rigenerazione delle risorse, energia e mortalità. Ogni individuo ha:
- 12 neuroni ricorrenti tanh, 15 sensori locali inclusi pareti/organismi vicini, odore direzionale del cibo, energia e feedback;
- 5 azioni controllate dai propri pesi: quattro mosse e **RIPRODURSI**;
- energia iniziale 36, metabolismo 1 per azione, cibo +18 (tetto 54), morte per fame o massimo 100 tick d'età;
- facoltà di riprodursi soltanto se età ≥4, energia ≥34, uno spazio adiacente libero e capienza popolazione non esaurita. Ogni nuova vita è un **organismo distinto**: il genitore paga 20 energia oltre al metabolismo e il figlio riceve 12 energia. Le altre 8 vengono dissipate: **non si crea energia dal nulla**.
- genoma numerico di **401 pesi**: al concepimento viene ereditato e alterato con rumore gaussiano SD 0,12 in ogni peso (braccio primario), copiato esattamente oppure sostituito con valori indipendenti.

Il cibo rigenera tramite due tentativi per tick, probabilità 0,55 ciascuno (ambiente principale), capacità massima 32 cibi; tutte le posizioni candidate sono sorteggiate **prima** della simulazione e sono identiche fra trattamenti dello stesso seed. Il figlio agisce solo dal tick seguente. Ogni evento di nascita registra genitore, discendenza, generazione e digest del genoma.

Il codice mantiene genealogie e bilancio energetico chiuso; il mondo **non cambia le leggi** per premiare genomi in modo esterno. Ma energia, regole di nascita, sensori, azioni e mutazione restano **disegnati da noi**: il confronto è un esperimento di evoluzione artificiale, non vita autonoma illimitata.

## 2. Test principale — 12 mondi indipendenti, 400 tick

Tutti i bracci partono **dalle stesse 20 reti fondatrici**, dalla stessa mappa e da un identico programma di tentativi di rigenerazione alimentare.

| Braccio | Nascite medie per mondo | Cibo consumato medio | Generazione massima media | Popolazione finale |
|---|---:|---:|---:|---:|
| **Genoma ereditato + mutazioni** | **23,42** | **70,00** | **3,00** | **0** |
| **Genoma completamente casuale per neonato** | **20,83** | **60,92** | **2,50** | **0** |
| **Genoma ereditato identico, senza mutazione** | **20,17** | **59,17** | **2,83** | **0** |

### Confronti preregistrati appaiati a livello di mondo

| Confronto nascita media | Differenza | Bootstrap descrittivo 95% | Segni per mondo |
|---|---:|---:|---|
| **Eredità mutata − figlio casuale (PRIMARIO)** | **+2,58** | **−2,58 / +9,17** | 6 vantaggi, 5 svantaggi, 1 pari |
| **Eredità mutata − clone senza mutazione** | +3,25 | **−2,00 / +9,25** | 6 vantaggi, 3 svantaggi, 3 pari |

Il protocollo richiedeva un vantaggio medio positivo, limite inferiore bootstrap positivo, **almeno nove mondi favorevoli su dodici**, e una generazione ≥3 in **almeno nove dei dodici mondi**.

**Entrambi i gate falliscono.** Nascite con ereditarietà mutata sono più numerose mediamente, ma l'evidenza non è robusta e la profondità ≥3 compare soltanto in **6/12 mondi**.

Inoltre il mondo **71002 è un forte outlier**: 67 nascite ereditarie mutate contro 33 casuali, **+34**; da solo contribuisce più dell'intero vantaggio aggregato (+31 su 12 mondi). Togliendolo, il confronto descrittivo passa da +2,58 a **−0,27** nascite/mondo. Questa analisi è *post-hoc* e serve solo a mostrare fragilità del segnale, non a ridefinire il primario.

Il mondo 71002 raggiunge **ottava generazione** — fatto interessante, ma un singolo caso fortunato **non prova adattamento riproducibile**.

## 3. Una scoperta concreta: le popolazioni muoiono con il cibo ancora disponibile

Le 12 popolazioni ereditarie mutate si estinguono **tutte** prima della fine dei 400 tick. Il momento medio di estinzione è **146,9** (massimo **286**), contro **130,6** del random-offspring e **126,5** del clone. Non è stato preregistrato un test di significatività separato per questi tempi: non rivendichiamo una vittoria sulla durata.

### Dinamica media main

| Tick | Organismi vivi eredità+mutazione | Figli casuali | Cloni | Cibo presente eredità+mutazione (max 32) |
|---:|---:|---:|---:|---:|
| 0 | 20,00 | 20,00 | 20,00 | 20,00 |
| 50 | 11,25 | 11,67 | 11,42 | 19,67 |
| 100 | **2,92** | 1,75 | 2,17 | **31,33** |
| 150 | 1,17 | 0,42 | 0,58 | 31,17 |
| 200 | 1,50 | 0,08 | 0,00 | ~32 |
| 300 | 0,00 | 0,00 | 0,00 | 32 |
| 400 | **0,00** | **0,00** | **0,00** | 32 |

Il fatto che al tick 100 il cibo sia quasi al massimo della capacità mentre gli organismi stanno scomparendo suggerisce un limite di **foraggiamento/accesso alle risorse e controllo motorio**, più che una semplice insufficienza di rigenerazione globale. Non dimostra quale componente fallisca: la rete potrebbe oscillare, concentrarsi su riproduzione, esplorare male o disperdere energia. Per distinguerli serve un esperimento comportamentale dedicato.

Il motore registra moltissime preferenze di azione che vengono corrette dal mascheramento: in media **869,9 correzioni per mondo** nel braccio eredità+mutazione, e **322,3 desideri di riprodursi quando le condizioni fisiche lo impediscono**. Sono valori cumulati su molti individui e tick, **non** tentativi per singolo organismo.

## 4. Ereditarietà reale, ma non vantaggio evolutivo accertato

Le genealogie e le misure genetiche mostrano che:
- il coefficiente coseno medio **genitore–figlio** vale circa **0,967** nel braccio mutato;
- vale esattamente **1,000** nei cloni senza mutazione;
- vale circa **0,001** nei figli con genoma indipendente.

Questa è una verifica concreta della **trasmissione genetica e dell'intervento di controllo**, non una misura di miglior fitness biologico. Sono stati osservati nuovi organismi, trasmissione parentale, mutazioni e fino a otto generazioni, senza l'intervento di un selezionatore esterno dei genomi.

Tuttavia la selezione non ha ancora prodotto un miglioramento **stabile, generalizzabile e indipendente dal mondo**. Tutte le linee, anche le più longeve, si sono estinte entro il tick 286 su 400.

## 5. Stress test: mondo più ostile (8 seed indipendenti)

Obiettivi evolutivi identici, ma nuovi mondi con muri più densi, odore limitato a 2 celle e rigenerazione per tentativo ridotta da 0,55 a 0,40.

| Braccio | Nascite medie | Cibo consumato | Generazione massima media | Popolazione finale |
|---|---:|---:|---:|---:|
| Genoma mutato ed ereditato | **17,13** | 43,25 | 2,00 | **0** |
| Genoma casuale nei neonati | **18,00** | 51,00 | 2,25 | **0** |
| Clone senza mutazione | **16,75** | 40,75 | 2,00 | **0** |

**Primario trasferito: −0,88 nascite/mondo**, bootstrap descrittivo **−4,50/+3,38**, **2 mondi positivi e 6 negativi**. Anche qui **tutte le popolazioni si estinguono**; l'ereditarietà mutata non ha dimostrato superiorità.

Non si trasferiscono genomi addestrati sui mondi principali: ogni mondo inizia dal proprio pool di fondatori e **evolve al proprio interno**. Il trasferimento qui riguarda le regole del meccanismo evolutivo, non un singolo genoma congelato.

## 6. Controlli fisici e validità dei risultati

- **Quattro mondi senza cibo e rigenerazione**: tutte le 20 fondatrici di ciascun mondo muoiono esattamente al tick 36; **nessuna nascita, nessuna risorsa raccolta**. Il meccanismo energetico dipende realmente dalle risorse.
- **Bilancio energetico numerico**: residuo esattamente **0,0** per tutte le misure archiviate: energia iniziale + energia effettiva da cibo = metabolismo + dissipazione da riproduzione + energia scartata alla morte per età + energia degli organismi ancora vivi.
- Ogni tratto con fondatori dello stesso seed nasce dallo **stesso vettore di genomi e posizioni**, indipendentemente dal meccanismo ereditario successivo.
- Posizioni alimentari iniziali raggiungibili, corpi non sovrapposti, assenza di attraversamento delle pareti, limiti energetici e demografici rispettati.
- Le nascite sono **atti eseguiti dalla rete in situazione di energia disponibile**, non cloni inseriti da un ottimizzatore esterno. Tuttavia la regola fisica che abilita la nascita è esplicita nel simulatore.

**Vincolo statistico:** gli eventi di nascita dentro uno stesso mondo condividono ambiente e linee genealogiche; **non sono 281 osservazioni indipendenti**. Le sole 12 repliche del primario sono gli interi mondi. Alcuni confronti secondari non hanno una correzione multipla; nessuno sostituisce il gate principale.

## 7. Stato scientifico e come procedere

**Successo ingegneristico:** abbiamo un ecosistema simulato persistente con genealogia, genomi ereditati, mutazioni, nascita per azione scelta dall'organismo, competizione ecologica, risorse rinnovabili e morte. Un algoritmo esterno **non** assegna un punteggio agli individui per selezionare il campione vincente.

**Insuccesso scientifico sul primario:** non c'è evidenza robusta che l'ereditarietà con nuove mutazioni aumenti la natalità rispetto ai figli casuali o rispetto ai cloni. Più importante, l'ambiente **non** mantiene ancora una popolazione viva. La rete neurale non appare capace di sfruttare abbastanza le risorse ambientali.

### M5.1 — proposta con controlli prima di modificare l'ecosistema

1. **Test di fattibilità ecologica:** programmare per scopi di confronto (NON dare questo programma alla rete) un foraggiatore semplice che segue l'odore, evita i muri e si riproduce quando energeticamente possibile. Se anche questa policy si estingue, problema di struttura delle risorse/energetica; se sopravvive, il collo di bottiglia è plausibilmente il controllo neurale e/o l'evoluzione.
2. **Diagnosi causale dei comportamenti:** registrare frazione di azioni motorie valide, cibo raccolto per 100 mosse, energia per età, tempo passato vicino al cibo, tentativi di riproduzione in stato impossibile, dispersione spaziale e durata delle linee. Confrontare contro un agente che si muove casualmente ma è obbligato a riprodursi energeticamente, senza presentarlo come evoluzione.
3. **Genoma ed ereditarietà:** testare, su **nuovi seed e protocollo congelato**, mutazioni sparse/piccole e possibilmente parametri evolvibili di sensibilità al cibo; separare effetti di mutazioni e selezione di caratteristiche già presenti.
4. **Prospettiva ecologica:** solo se ci sono popolazioni non estinte su test nuovi si possono introdurre risorse variabili, ciclo vita/energia più articolato, competizione e, in seguito, predatori. Non correggere i dati storici M5.0 aumentando il cibo sugli stessi seed.
5. **Un criterio di successo più forte:** oltre al conteggio nascite, richiedere persistenza di popolazioni, miglioramento energetico per discendente in generazioni diverse e un vantaggio contro controlli non ereditari. Non basta contare generazioni.

Non alterare i parametri, seed o conclusioni di M5.0 a posteriori. I prossimi esperimenti devono essere considerati **M5.1**, con nuovi seed e gate fissati prima della campagna.

**Nessuna rivendicazione di organismo biologico o coscienza:** metabolismo, sistema riproduttivo, sensori, tipi di azione e mutazione sono progettati dall'essere umano. La selezione ecologica è un meccanismo reale della simulazione, ma non dimostra l'origine autonoma della vita o dell'intelligenza.
