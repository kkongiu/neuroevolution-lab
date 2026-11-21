# M3.4 — Riconoscere il contesto da esempi osservati: risultati della prima campagna

**Data:** 21 novembre 2026.  
**Esito principale:** **positivo SOLO nel benchmark con 16 esempi di supporto puliti e famiglie note**.  
**Trasferimento:** parziale; riconoscimento robusto di un contesto che ritorna **non confermato** in condizioni di pochi indizi rumorosi e nuove famiglie.  
**Replay:** miglioramenti numerici interessanti, ma non confermati statisticamente sul criterio preregistrato.

- [Protocollo congelato prima dell'implementazione e dei test finali](../docs/M3_4_PROTOCOL.md)
- [Codice M3.4](../neuroevolution/m3_4.py), [pull request #15](https://github.com/kkongiu/neuroevolution-lab/pull/15), integrata in main.
- [GitHub Actions #44901460644](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44901460644) — **success**, con tutti i test.
- [CSV per ciascuno dei 48 seed](M3_4_seed_metrics_2026-11-21.csv); valori principali arrotondati a sei decimali.
- **Artefatto originale integrale:** GitHub Actions **m3-4-latent-context-inference**, ID **14706072325**, scadenza prevista 21 dicembre 2026. Contiene JSON completo, 240 righe seed×braccio, 2.880 righe episodio×braccio e 48 righe di qualità dell'inferenza.

## 1. Cosa abbiamo effettivamente costruito

In M3.3 la rete riceveva il contesto corretto come ID a quattro bit. In M3.4 l'ID è **nascosto** al sistema di decisione: ciascun episodio offre solo una piccola sequenza di **punti x,y con feedback (etichetta)**.

Un riconoscitore k-nearest-neighbours scritto da noi — **non appreso spontaneamente dalla rete** — confronta quei punti con esempi TRAINING conservati dai contesti precedentemente incontrati. Se la corrispondenza migliore supera una soglia, riutilizza l'identificativo anonimo della vecchia memoria; altrimenti, se possibile, crea un nuovo slot. La rete ReLU persistente riceve x,y e i quattro bit di quello *slot inferito*, non l'ID latente dell'esperimento.

Il numero reale del contesto compare soltanto nel simulatore che genera le etichette e nell'**audit retrospettivo** delle decisioni. La verifica automatica altera gli ID reali e rimuove i pannelli di valutazione: le decisioni del riconoscitore restano identiche.

Gli esempi di supporto sono **etichettati** e provenienti da un flusso diverso sia dagli esempi SGD sia dal holdout. Non abbiamo mostrato che un sistema possa identificare regole opposte senza indizi o senza feedback.

### Separazione dati e budget

- Meta-training: **6 seed 24000–24005**; meta-validation: **3 seed 25000–25002**.
- Test principale: **24 seed 26000–26023**, regole line/XOR con 16 supporti **corretti** per episodio.
- Test pochi indizi: **12 seed 27000–27011**, stesse famiglie, **6 supporti e un'etichetta alterata**.
- Test famiglia nuova: **12 seed 28000–28011**, strisce/scacchiera, stessi sei supporti e un errore.
- **4 contesti, 12 episodi, un'unica rete persistente**, 24 unità ReLU; 48 update SGD×batch 4/episodio, stesso budget per tutte le reti.
- Fino a **4 × 64 = 256 esempi TRAINING** come prototipi per identificare il contesto.
- Replay separato fino a **4 × 24 = 96 esempi TRAINING**: un mini-batch su quattro viene **sostituito** con un batch vecchio, non aggiunto. Il recupero ha quindi un costo in esempi del task corrente.
- Risoluzione kNN **k=7**. La soglia di ammissione **0,84** è stata scelta prima del test finale fra i valori [0,60, 0,68, 0,76, 0,84], con meta-train/meta-validation. Su entrambi quei gruppi il punteggio di raggruppamento della soglia selezionata era **100%**. Il mondo familiare è quindi favorevole a questo algoritmo.

### Controlli

Cinque traiettorie di rete per seed:
1. contesto inferito + replay FIFO;
2. contesto inferito + replay reservoir (stessa capienza);
3. contesto inferito + nessun replay;
4. **ID reale del contesto** per una rete di riferimento privilegiata;
5. rete con bit di contesto azzerati, ma **stesso raggruppamento inferito per il replay**.

Per la traiettoria 1 valutiamo in aggiunta **senza riaddestrarla** gli stessi pesi con i supporti finali rimescolati e con nessun contesto. Il controllo rimescolato ruota le quattro sequenze di indizi: può portare volutamente a identificativi incompatibili e a prestazioni **inferiori al caso**. È un intervento sul segnale, non una baseline casuale generalizzabile.

## 2. Risultato principale — mondo familiare, 24 seed

**Accuratezza finale media bilanciata sui quattro contesti:**

| Strategia | Accuratezza |
|---|---:|
| **Contesto inferito + replay FIFO** | **72,18%** |
| Stessa rete, supporti finali scambiati (primario) | 38,95% |
| Stessa rete, nessun supporto/contesto | 50,00% |
| Contesto inferito, nessun replay | 69,21% |
| Contesto inferito, replay reservoir | 70,04% |
| Rete con ID reale di riferimento | 72,18% |
| Rete senza contesto, con replay | 50,00% |

**Confronto primario preregistrato** (inferred FIFO meno la stessa rete con supporti rimescolati):

- **+33,24 punti percentuali** (differenza appaiata).
- Bootstrap percentile descrittivo 95% su 24 seed: **+28,74 / +37,50 pp**.
- **24/24 seed positivi**.

**Confronto necessario con zero contesto**: **+22,18 pp**, intervallo **+19,35 / +24,82 pp**, tutti i 24 seed positivi. La baseline senza contesto non può andare oltre il 50% **per costruzione matematica**: le coppie di task assegnano etichette opposte ai medesimi x,y. Non è una baseline difficile.

### Ha riconosciuto effettivamente il ritorno dei vecchi contesti?

Sì, su questi dati facili:
- **4,00 / 4** slot distinti nei primi quattro contesti nuovi, in media.
- **100% di assegnazioni coerenti al primo slot** quando ritorna la stessa regola.
- **100% di coerenza** per l'assegnazione di nuovi supporti indipendenti a fine sequenza.
- **100% di balanced pairwise context-clustering**.

Questo soddisfa tutti i tre gate fissati per il test principale: vantaggio sui supporti scambiati, valore sopra il 50% senza supporto, coerenza dei ritorni >65% e distinzione dei contesti iniziali >=3,5. **È tuttavia un successo in un mondo sintetico già facilmente discriminabile con 16 supporti corretti.**

La rete privilegiata con ID reali raggiunge **la stessa accuratezza media** 72,18%, ma non gli stessi esiti per ogni seed: l'ordine anonimo degli slot modifica l'associazione con i pesi iniziali. La differenza media è 0,00 pp, bootstrap **−2,33 / +2,31 pp**. L'oracolo è una *referenza con informazioni privilegiate*, non un limite superiore garantito per ogni singola inizializzazione.

L'accuratezza media del **contesto peggiore** è soltanto **49,54%** con replay FIFO: la capacità di riconoscere il contesto non implica che la rete mantenga perfettamente tutte le conoscenze.

## 3. Quando gli indizi diventano incompleti o inesatti

| Misura | Principale: 16 supporti corretti | 6 supporti + 1 errore | Nuove famiglie + supporto difficile |
|---|---:|---:|---:|
| **Accuratezza finale — contesto inferito FIFO** | **72,18%** | **66,37%** | **54,30%** |
| Accuracy con supporti scambiati | 38,95% | 40,10% | 49,22% |
| Accuratezza con ID reale | 72,18% | 74,32% | 59,64% |
| Slot distinti nelle prime quattro visite | 4,00 | 3,83 | 3,92 |
| **Ritorni assegnati al primo slot corretto** | **100%** | **62,50%** | **30,21%** |
| Coerenza di nuovi indizi finali | 100% | 50% | 25% |
| Qualità bilanciata della distinzione fra episodi | 100% | 76,19% | 58,18% |

### Test con pochi supporti, stesse famiglie (12 seed)

- Differenza inferred − supporti scambiati: **+26,27 pp**, bootstrap **+19,34 / +32,42 pp**, 11 seed positivi e uno negativo.
- Differenza inferred − zero supporto: **+16,37 pp**, tutti i 12 seed positivi.
- Differenza inferred − oracle ID: **−7,94 pp** in media, bootstrap **−11,00 / −5,01 pp**.
- Ritorno del contesto al primo slot: **62,5%**, **sotto la soglia preregistrata del 65%**.

### Test su famiglie nuove E supporto rumoroso (12 seed)

- Differenza inferred − supporti scambiati: **+5,08 pp**, bootstrap **+0,85 / +10,32 pp**, 9 seed positivi e 3 negativi.
- Differenza inferred − zero supporto: **+4,30 pp**, bootstrap **+1,20 / +7,59 pp**, 9 positivi e 3 negativi.
- Differenza inferred − oracle ID: **−5,34 pp** in media, bootstrap **−9,57 / −0,81 pp**.
- Ritorno del contesto al primo slot: **30,21%**, nettamente sotto la soglia richiesta.
- Coerenza dell'inferenza da supporti mai incontrati: soltanto **25%**.

**Conclusione di trasferimento:** anche quando l'accuratezza finale supera debolmente la baseline priva di contesto, non possiamo parlare di riconoscimento affidabile e generalizzabile. Su nuove famiglie il resolver sbaglia spesso identità; alcuni contesti sono difficili anche per la rete che riceve l'ID reale (59,64% di accuratezza). Questi due limiti devono restare distinti.

## 4. Replay FIFO, reservoir, o nessun replay?

Nel mondo principale, a parità di architettura, inferenza, SGD e memoria allocabile:

| Metodo | Accuratezza finale | AUC mentre si apprende un episodio | Dimenticanza dal picco alla fine |
|---|---:|---:|---:|
| **FIFO + replay** | **72,18%** | 68,11% | 10,76 pp |
| Reservoir + replay | 70,04% | 67,68% | 11,91 pp |
| Nessun replay | 69,21% | **69,42%** | 14,10 pp |

- FIFO − nessun replay: **+2,98 pp** sulla memoria finale; intervallo bootstrap **−0,02 / +6,05 pp** (14 seed positivi, 9 negativi, 1 pari). **Non confermato** dal nostro intervallo perché comprende zero.
- FIFO − reservoir: **+2,15 pp**, intervallo **−0,80 / +5,09 pp** (14 positivi, 10 negativi): **non confermato**.
- Il replay migliora numericamente la ritenzione ma sottrae un quarto dei batch al task corrente; non è gratis. La minore AUC per episodio rispetto a nessun replay è coerente con questo compromesso.
- Sul test con famiglie nuove, FIFO **54,30%** contro no replay **54,49%**, differenza **−0,20 pp** con intervallo che comprende zero. **Nessun beneficio di replay dimostrato** fuori distribuzione.
- Il reservoir è **numericamente migliore** di FIFO sul test con famiglie nuove (56,61% contro 54,30%), ma la differenza ha intervallo incerto: non è una vittoria robusta né una ragione per retro-modificare la regola scelta.

## 5. Audit tecnico e limiti

1. **Il riconoscitore è programmato da noi.** È kNN con una soglia scelta offline sui veri ID latenti soltanto in meta-training/meta-validazione. La rete persistente usa l'ID anonimo derivato ma NON inventa l'algoritmo di riconoscimento.
2. **Esistono etichette dei supporti.** Non stiamo inferendo il contesto da input completamente non etichettati. Questa è una forma di esperienza supervisionata.
3. **Il problema principale è facile per il resolver.** La meta-validazione e tutti i 24 test familiari arrivano al 100% di clustering; non è prova di capacità generali, quanto piuttosto di sufficiente separabilità delle regole date le informazioni.
4. **Il controllo con supporti ruotati è intenzionalmente avverso.** Ha accuratezza **38,95%** sul main, sotto 50%: scambiare una regola con una invertita può produrre sistematicamente risposte sbagliate, amplificando il contrasto. Anche il controllo zero supporto è bloccato al 50% per la struttura di etichette opposte.
5. **La validità del trasferimento peggiora drasticamente.** La metrica di coerenza di ritorno fallisce il gate (62,5% e 30,21%); il miglioramento sulla predizione è molto più debole della prova di identità.
6. **Replay e prototipi hanno risorse proprie.** Fino a 256 esempi di prototipi più 96 esempi di replay per seed. Il test familiare richiede circa **7.840 valutazioni di vicini kNN** per sequenza, oltre agli update SGD; non abbiamo dimostrato parità totale di CPU/energia.
7. **Pochi supporti e nuove famiglie cambiano due fattori.** Abbiamo un test intermedio stessi problemi+supporto rumoroso, che isola *parzialmente* la ragione del degrado. I task shifted sono comunque intrinsecamente più difficili per il modello.
8. Le differenze e gli intervalli sono **descrittivi appaiati al livello del seed** (24 o 12). Non trattare le 576 osservazioni di episodio per le 48 sequenze o i 64 punti holdout come repliche indipendenti. Confronti secondari multipli non sono corretti per testing simultaneo.
9. Nessuna auto-riscrittura del codice, né obiettivi propri né un sistema in vita aperta. Non ci sono accessi esterni concessi ai modelli simulati.

## 6. Conclusione e prossima ipotesi

**M3.4 è riuscita nel mondo semplice preregistrato**: un sistema può associare con precisione una sequenza di indizi etichettati a esperienze precedenti, richiamando quindi il contesto senza ricevere l'ID reale. Il risultato è operativo e riproducibile.

**Non è ancora robusta.** Il passaggio più informativo è che, su problemi nuovi, la difficoltà principale si sposta dal solo apprendimento dei pesi al **riconoscimento dell'identità del contesto** e alla possibilità di riorganizzare un insieme limitato di ricordi senza confonderli.

Per M3.5 proporre *prima* un confronto su stessi contesti ma diversi livelli di rumore in cui il sistema abbia la possibilità di chiedere **un esempio aggiuntivo solo quando è incerto**, a budget totale di esempi fisso e con una baseline a richieste casuali/periodiche. Distinguere:
- riconoscimento della novità;
- richiesta informativa attiva;
- utilità dei nuovi dati sulla memoria e accuratezza predittiva;
- costo in indizi e tempo computazionale.

È il primo passo sensato verso un'organizzazione della propria esperienza, senza confondere un algoritmo di retrieval che abbiamo scritto noi con un'intelligenza che inventa autonomamente le proprie capacità.

**Non scegliere nuove soglie o modificare M3.4 dopo aver visto i 48 seed finali.** Eventuali cambiamenti richiedono una nuova milestone con seed e protocollo propri.

## Provenienza e riproducibilità

[GitHub Actions #44901460644](https://github.com/kkongiu/neuroevolution-lab/actions/runs/44901460644) conserva per 30 giorni i quattro file:
- \`summary.json\`: parametri e risultati, anche gli intervalli bootstrap per seed.
- \`seed_metrics.csv\`: **240 righe** di accuratezza, ritenzione e AUC per braccio e seed.
- \`episode_metrics.csv\`: **2.880 episodi** dei cinque trattamenti.
- \`inference_metrics.csv\`: **48 righe** di purezza/consistenza del riconoscitore.

Questo report e il [CSV sintetico per i 48 seed](M3_4_seed_metrics_2026-11-21.csv) sono conservati in modo permanente nel repository, dopo la scadenza dell'artefatto integrale.
