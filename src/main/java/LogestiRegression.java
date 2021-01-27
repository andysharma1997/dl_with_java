import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class LogestiRegression {
    private static final Logger logger = LogManager.getLogger(LogestiRegression.class);
    public static String dataLocalPath;
    public static Integer bach_size=50;
    double learning_rate = 0.01;
    public  static Integer n_epocs = 30;
    public static Integer n_inputs = 21;
    public static Integer n_outputs = 2;
    public static Integer n_hidden = 20;

    private static HashMap<String, DataSet> load_data(String file_name) throws IOException, InterruptedException {
       RecordReader recordReader = new CSVRecordReader(1,",");
       ClassPathResource resource = new ClassPathResource(file_name);
       recordReader.initialize(new FileSplit(resource.getFile()));
       DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, bach_size,n_inputs,n_outputs);
        DataSet all_data = iterator.next();
        all_data.shuffle(123);
        SplitTestAndTrain test_and_train = all_data.splitTestAndTrain(0.80);
        DataSet train_data = test_and_train.getTrain();
        DataSet test_data = test_and_train.getTest();
        HashMap<String,DataSet> test_train_data = new HashMap<>();
        test_train_data.put("train_data",train_data) ;
        test_train_data.put("test_data",test_data);
        return test_train_data;
    }

    private static void model(DataSet train,DataSet test){
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01,0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(n_inputs).nOut(n_hidden)
                        .activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(n_hidden).nOut(n_outputs).build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        for (int epoch=0;epoch<n_epocs;epoch++){
            model.fit(train);
        }
        System.out.println("Evaluating model....");
        Evaluation eval = new Evaluation(n_outputs);
        while (test.hasNext()){
            DataSet t = test.next();
            INDArray features = t.getFeatures();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);
            eval.eval(lables,predicted);
        }
        System.out.println(eval.stats());
    }

    public static void main(String[] args) {

    }
}
