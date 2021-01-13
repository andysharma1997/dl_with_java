import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class IrishClassification {
    private static final Logger logger = LogManager.getLogger(IrishClassification.class);
    private static int FEATURE_COUNT = 4;
    private static int CLASS_COUNT = 3;
    public static void main(String[] args) {
        BasicConfigurator.configure();
        load_data();

    }
    private static void load_data(){
        try(RecordReader recordReader = new CSVRecordReader(1,',')){
            ClassPathResource resource = new ClassPathResource("IRIS.csv");
            logger.info("Reading file= "+resource.getFilename());


            recordReader.initialize(new FileSplit(resource.getFile()));
            logger.info("Creating bach iterator");
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,150,FEATURE_COUNT,CLASS_COUNT);
            logger.info(iterator);
            DataSet all_data = iterator.next();

            all_data.shuffle(123);
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(all_data);
            normalizer.transform(all_data);
            SplitTestAndTrain test_and_train = all_data.splitTestAndTrain(0.65);
            DataSet train_data = test_and_train.getTrain();
            DataSet test_data = test_and_train.getTest();
            irishNetwork(train_data,test_data);
        }catch (Exception e){
            System.out.println("Error: "+e.getLocalizedMessage());
        }
    }

    private static void irishNetwork(DataSet train_data,DataSet test_data){
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.1, 0.9))
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURE_COUNT).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                .layer(2, new OutputLayer.Builder(
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASS_COUNT).build())
                .backpropType(BackpropType.Standard)
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.fit(train_data);
        INDArray output = model.output(test_data.getFeaturesMaskArray());
        Evaluation eval = new Evaluation();
        eval.eval(test_data.getLabels(),output);
        System.out.println(eval.stats());
    }
}
