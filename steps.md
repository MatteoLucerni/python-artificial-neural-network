# Rete Neurale Artificiale (ANN) a Percettrone Multistrato (MLP)

## Passaggio 1: caricamento assets

    Carichiamo tutti gli strumenti necessari, tra cui librerie di Data Science e i dati che vogliamo passare come input

## Passaggio 2: Preprocessing dei dati

    1. Suddivisione del dataset in due set X e Y, dove X saranno le proprietà passate in analisi e Y sarà l'etichetta (classe di appartenza)

    2. Ulteriore divisione di X e Y in subsets di TRAIN e di TEST

    3. Standardizzazione dei valori delle proprietà delle immagini in modo da averle in una scala ristretta da 0 a 1 che sia addatta alla rete neurale

## Passaggio 3: Dichiarazione, Addestramento e Predizione

    1. Definisco la struttura della rete neurale scegliendo gli iperparametri più adatti

    2. Addestro il modello con i dati di TRAIN di X e Y

    3. Gli faccio eseguire delle predizioni su i dati di TEST di X

## Passaggio 4: Validazione e valutazione

    1. Valuto l'accuraztezza e la loss conforntando le predizioni eseguite sul set di TEST di X con le effettive etichette del relativo set di TEST di Y

    3. Per validarne il risultato gli faccio eseguire delle predizioni anche sui dati di TRAIN di X calcolando accuratezza e loss grazie alle etichette di TRAIN di Y
