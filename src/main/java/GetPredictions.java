import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.IOException;

public class GetPredictions {
    public static void main(String[] args) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("/home/andy/git/dl_with_java/src/main/resources/won_lost_model");
        System.out.println(model.getLayerWiseConfigurations());
    }
}
