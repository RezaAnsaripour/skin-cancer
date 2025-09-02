clc
clear all
close all
warning off;
delete(gcp('nocreate'))
m=2;
switch m
    case 1
        str="HAM10000";
        label= importham("HAM10000", [2, Inf]);
        label2=label.dx;
        image_idx=label.image_id;
        [image_idx,sidex]=sort(image_idx,"ascend");
        label2=label2(sidex);
        imds = imageDatastore("HAM1000\", ...
            IncludeSubfolders=false, ...
            Labels=label2);
        imds=subset(imds,1:100);
    case 2
        str="ISIC_2019";
        ISIC2019 = importISIC2019("ISIC_2019.csv", [2, Inf]);
        class=ISIC2019.Properties.VariableNames;
        for i=2: width(ISIC2019)
            idx=table2array(ISIC2019(:,i));
            idx=(idx==1);
            label(idx,1)=class(i);
        end

        label2=categorical(label);
        image_idx=ISIC2019.image;
        [image_idx,sidex]=sort(image_idx,"ascend");
        label2=label2(sidex);
        imds = imageDatastore("ISIC_2019\", ...
            IncludeSubfolders=false, ...
            Labels=label2);
        imds=subset(imds,1:100);
end

%% FDL
cluster = parcluster("Processes");
cluster.NumWorkers = 5;
pool = parpool(cluster);
numWorkers = pool.NumWorkers;
inputSize = [100 100 3];
spmd
    [imdsTrain,imdsTestVal] = splitEachLabel(imds,0.8,"randomized");

    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
end

fileList = [];
labelList = [];

for i = 1:numWorkers
    tmp = imdsTestVal{i};

    fileList = cat(1,fileList,tmp.Files);
    labelList = cat(1,labelList,tmp.Labels);
end

imdsGlobalTestVal = imageDatastore(fileList);
imdsGlobalTestVal.Labels = labelList;

[imdsGlobalTest,imdsGlobalVal] = splitEachLabel(imdsGlobalTestVal,0.5,"randomized");

augimdsGlobalTest = augmentedImageDatastore(inputSize(1:2),imdsGlobalTest);
augimdsGlobalVal = augmentedImageDatastore(inputSize(1:2),imdsGlobalVal);

classes = categories(imdsGlobalTest.Labels);
numClasses = numel(classes);
l=2;
switch l
    case 1  % simple ShuffleNet block in CNN
        layers = [
            imageInputLayer(inputSize)

            convolution2dLayer(5,32,'Padding','same')
            reluLayer

            groupedConvolution2dLayer(5,1,'channel-wise','Stride',2,'Padding','same')
            reluLayer
            
            convolution2dLayer(5,64,'Padding','same')
            reluLayer

            maxPooling2dLayer(2,'Stride',2)

            fullyConnectedLayer(numClasses)
            softmaxLayer];

         net = dlnetwork(layers)
         learnRate = 0.00001;


    case 2   %CBAM CNN
        layers = [
            imageInputLayer(inputSize,"Normalization","none","Name","input")
            convolution2dLayer(5,32,"Name","conv1")
            reluLayer("Name","relu1")
            maxPooling2dLayer(2,"Name","maxpool1")
            convolution2dLayer(5,64,"Name","conv2")
            reluLayer("Name","relu2")
            ];

        lgraph = layerGraph(layers);

        avgPool = averagePooling2dLayer([1 1],"Name","avgpool_cbam","Padding","same");
        maxPool = maxPooling2dLayer([1 1],"Name","maxpool_cbam","Padding","same");
        multiply = multiplicationLayer(2,"Name","channel_multiply");
        reluAtt = leakyReluLayer(0.01,"Name","cbam_relu");
        sigmoidAtt = sigmoidLayer("Name","cbam_sigmoid");
        lgraph = addLayers(lgraph, avgPool);
        lgraph = addLayers(lgraph, maxPool);
        lgraph = addLayers(lgraph, multiply);
        lgraph = addLayers(lgraph, reluAtt);
        lgraph = addLayers(lgraph, sigmoidAtt);

        lgraph = connectLayers(lgraph,"relu2","avgpool_cbam");
        lgraph = connectLayers(lgraph,"relu2","maxpool_cbam");
        lgraph = connectLayers(lgraph,"avgpool_cbam","channel_multiply/in1");
        lgraph = connectLayers(lgraph,"maxpool_cbam","channel_multiply/in2");
        lgraph = connectLayers(lgraph,"channel_multiply","cbam_relu");
        lgraph = connectLayers(lgraph,"cbam_relu","cbam_sigmoid");
        finalayers=[fullyConnectedLayer(numClasses,"Name","fc")
            softmaxLayer("Name","softmax")
            ];
        lgraph = addLayers(lgraph,finalayers);
        channelAttentionMult = multiplicationLayer(2, "Name", "attention_multiply");
        lgraph = addLayers(lgraph, channelAttentionMult);
        lgraph = connectLayers(lgraph,"relu2","attention_multiply/in1");
        lgraph = connectLayers(lgraph,"cbam_sigmoid","attention_multiply/in2");
        lgraph = connectLayers(lgraph,"attention_multiply","fc");

        net = dlnetwork(lgraph)
        learnRate = 0.00001;
end
numRounds = 300;
numEpochsperRound = 5;
miniBatchSize = 100;
momentum = 0;


preProcess = @(x,y)preprocessMiniBatch(x,y,classes);

spmd
    sizeOfLocalDataset = augimdsTrain.NumObservations;

    mbq = minibatchqueue(augimdsTrain, ...
        MiniBatchSize=miniBatchSize, ...
        MiniBatchFcn=preProcess, ...
        MiniBatchFormat=["SSCB",""]);
end


mbqGlobalVal = minibatchqueue(augimdsGlobalVal, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=preProcess, ...
    MiniBatchFormat=["SSCB",""]);

monitor = trainingProgressMonitor( ...
    Metrics="GlobalAccuracy", ...
    Info="CommunicationRound", ...
    XLabel="Communication Round");


velocity = [];

globalModel = net;

round = 0;
while round < numRounds && ~monitor.Stop

    round = round + 1;

    spmd
        % Send global updated parameters to each worker.
        net.Learnables.Value = globalModel.Learnables.Value;

        % Loop over epochs.
        for epoch = 1:numEpochsperRound
            % Shuffle data.
            shuffle(mbq);

            % Loop over mini-batches.
            while hasdata(mbq)

                % Read mini-batch of data.
                [X,T] = next(mbq);

                % Evaluate the model loss and gradients using dlfeval and the
                % modelLoss function.
                [loss,gradients] = dlfeval(@modelLoss,net,X,T);

                % Update the network parameters using the SGDM optimizer.
                [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);

            end
        end

        % Collect updated learnable parameters on each worker.
        workerLearnables = net.Learnables.Value;
    end

    % Find normalization factors for each worker based on ratio of data
    % processed on that worker.
    sizeOfAllDatasets = sum([sizeOfLocalDataset{:}]);
    normalizationFactor = [sizeOfLocalDataset{:}]/sizeOfAllDatasets;

    % Update the global model with new learnable parameters, normalized and
    % averaged across all workers.
    globalModel.Learnables.Value = federatedAveraging(workerLearnables,normalizationFactor);

    % Calculate the accuracy of the global model.
    accuracy = computeAccuracy(globalModel,mbqGlobalVal,classes);

    % Update the training progress monitor.
    recordMetrics(monitor,round,GlobalAccuracy=accuracy);
    updateInfo(monitor,CommunicationRound=round + " of " + numRounds);
    monitor.Progress = 100*round/numRounds;

end

spmd
    net.Learnables.Value = globalModel.Learnables.Value;
end


mbqGlobalTest = minibatchqueue(augimdsGlobalTest, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=preProcess, ...
    MiniBatchFormat="SSCB");

accuracy = computeAccuracy(globalModel,mbqGlobalTest,classes)
[~,YPred_All,YTest_All] = computeAccuracy2(globalModel,mbqGlobalTest,classes);
plotconfusion(YPred_All,YTest_All,str)
delete(gcp("nocreate"));




%% Help Function

function [loss,gradients] = modelLoss(net,X,T)

YPred = forward(net,X);

loss = crossentropy(YPred,T);
gradients = dlgradient(loss,net.Learnables);

end

function accuracy = computeAccuracy(net,mbq,classes)

correctPredictions = [];

shuffle(mbq);
while hasdata(mbq)

    [XTest,TTest] = next(mbq);

    TTest = onehotdecode(TTest,classes,1)';

    YPred = predict(net,XTest);
    YPred = onehotdecode(YPred,classes,1)';

    correctPredictions = [correctPredictions; YPred == TTest];
end

predSum = sum(correctPredictions);
accuracy = single(predSum./size(correctPredictions,1));

end

function [X,Y] = preprocessMiniBatch(XCell,YCell,classes)

% Concatenate.
X = cat(4,XCell{1:end});

% Extract label data from cell and concatenate.
Y = cat(2,YCell{1:end});

% One-hot encode labels.
Y = onehotencode(Y,1,ClassNames=classes);

end


function learnables = federatedAveraging(workerLearnables,normalizationFactor)

numWorkers = size(normalizationFactor,2);

% Initialize container for averaged learnables with same size as existing
% learnables. Use learnables of first worker network as an example.
exampleLearnables = workerLearnables{1};
learnables = cell(height(exampleLearnables),1);

for i = 1:height(learnables)
    learnables{i} = zeros(size(exampleLearnables{i}),"like",(exampleLearnables{i}));
end

% Add the normalized learnable parameters of all workers to
% calculate average values.
for i = 1:numWorkers
    tmp = workerLearnables{i};
    for values = 1:numel(learnables)
        learnables{values} = learnables{values} + normalizationFactor(i).*tmp{values};
    end
end

end


function [accuracy,YPred_All,YTest_All] = computeAccuracy2(net,mbq,classes)

correctPredictions = [];
YPred_All=[];
YTest_All=[];
shuffle(mbq);
while hasdata(mbq)

    [XTest,TTest] = next(mbq);

    TTest = onehotdecode(TTest,classes,1)';
    YTest_All=[ YTest_All;TTest];
    YPred = predict(net,XTest);
    YPred = onehotdecode(YPred,classes,1)';
    YPred_All=[YPred_All;YPred];
    correctPredictions = [correctPredictions; YPred == TTest];
end

predSum = sum(correctPredictions);
accuracy = single(predSum./size(correctPredictions,1));

end