% Coursework Specification:
% We are given 60000 images of varying labels, and given labels.
% We select 3 different types of images, pseudorandomely, using a seed of
% our student number.
% We make our own K-Nearest-Neighbour model with two different distance
% metrics.
% We use some images for training and some for testing.
% We then use 2 built in implementations, and compare our own to that.
% We compare the Accuracy, Confusion Matrix, and the Time Taken.
% Writing a one-page document on our findings.

% Load the dataset:
dataset = load("cifar-10-data.mat");
images = dataset.data;
labels = dataset.labels;
label_names = dataset.label_names;

% Generate random numbers to determine my classes:
studentID = 38700514;
rng(studentID, 'twister');
classes = randperm(10, 3); % [10, 1, 4]
% We are working with Trucks, Airplanes, and Cats.

% Extract our images and labels we are working with:
selectedData = [];
selectedLabels = [];
for i = 1:3
    % find function gets specific indices of an array that meet a condition:
    classIdx = find(labels+1 == classes(i)); 
    % concatenate the selected* arrays with the data found in the other arrays:
    selectedData = cat(1, selectedData, images(classIdx,:,:,:));
    selectedLabels = cat(1, selectedLabels, labels(classIdx));  
    
end

% Split Data in half between Testing and Training:
rng(studentID, 'twister');
training_index = randperm(18000, 9000);
all_indices = 1:18000;
% Set diff gets all the indices that aren't in the training index:
test_index = setdiff(all_indices, training_index);

trainingData = selectedData(training_index, :, :, :);
trainingLabels = selectedLabels(training_index); 
testingData = selectedData(test_index, :, :, :);
testingLabels = selectedLabels(test_index);    

trainingData = reshape(trainingData, 9000, 3072);
testingData = reshape(testingData, 9000, 3072);
testingLabels = double(testingLabels);
testingData = im2double(testingData);
trainingData = im2double(trainingData);

ourLabels = {'truck'; 'airplane'; 'cat'};


%% Model Training

% K-Nearest-Neighbour time!
k = 5; % The number of neighbours we will visit.
cosinePredictions = zeros(9000, 1);
euclideanPredictions = zeros(9000, 1);

% Euclidean Distance:
tic;  % Start timer.

for i = 1:9000
    % Make a matrix of duplicate images for our image to compare with:
    currentImage = repmat(testingData(i, :), [size(trainingData, 1), 1]);

    % Get L2 Distances:
    distances = abs(sum((trainingData - currentImage).^2, 2));

    % Get the 5 neighbour's labels:
    [~, index] = sort(distances); % '~' discards, like '_' in Python
    nearestNeighbors = trainingLabels(index(1:k));

    % Our prediction is the most occuring label:
    euclideanPredictions(i,1) = mode(nearestNeighbors);

    disp("Working! i = "+i)
end
euclidean_timetaken = toc  % Get the time elapsed.

% Confusion Matrix
figure
euclidean_confusionmatrix = confusionchart(testingLabels, euclideanPredictions)

knnl2accuracy = sum(testingLabels == euclideanPredictions) / 9000

% Cosine Distance:
tic; % Start timer again.

for i = 1:9000
    % Make a matrix of duplicate images for our image to compare with:
    currentImage = repmat(testingData(i, :), [size(trainingData, 1), 1]);

    % Calculate cosine distances from the test sample to all training samples
    % Get Dot Product:
    dotProducts = sum((currentImage .* trainingData), 2);

    % Get the Norms:
    % Square each item on the second dimension, add them up, then square root the sum:
    trainingNorm = sqrt(sum(trainingData.^2, 2));
    testingNorm = testingData(i);

    % Get Cosine Distance:
    % Perform element-wise division on the dotProducts:
    cosineDistance = 1 - (dotProducts ./ (trainingNorm * testingNorm)); 

    % Get the 5 neighbour's labels:
    [~, index] = sort(cosineDistance); % '~' discards, like '_' in Python
     nearestNeighbors = trainingLabels(index(1:k));

    % Our prediction is the most occuring label:
    cosinePredictions(i,1) = mode(nearestNeighbors);

    disp("Working! i = "+i)
end
cosine_timetaken = toc % Get the time elapsed.

% Confusion Matrix
figure(2)
cosine_confusionmatrix = confusionchart(testingLabels,cosinePredictions)
cosineaccuracy = sum(testingLabels == cosinePredictions) / 9000
%% Testing other algorithms and Comparison/Evaluation

% Decision Tree Model:
tic;
decisionTreeModel = fitctree(trainingData, trainingLabels);
treePredictions = predict(decisionTreeModel, testingData);
treeaccuracy = sum(treePredictions == testingLabels) / 9000
treePredictions = predict(decisionTreeModel, testingData);
tree_timetaken = toc

% SVM For Multiclass Model:
tic;
ecocModel = fitcecoc(trainingData, trainingLabels);
ecocPredictions = predict(ecocModel, testingData);
ecocaccuracy = sum(ecocPredictions == testingLabels) / length(testingLabels)
ecocPredictions = predict(ecocModel, testingData);
ecoc_timetaken = toc

save('cw1', 'classes','training_index','knnl2accuracy','euclidean_timetaken', 'euclidean_confusionmatrix','cosineaccuracy','cosine_timetaken','cosine_confusionmatrix')
