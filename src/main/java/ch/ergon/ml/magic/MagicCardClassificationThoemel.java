package ch.ergon.ml.magic;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Optional;
import java.util.Random;

import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.common.RecordConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import lombok.NonNull;

/**
 * Animal Classification
 *
 * Example classification of photos from 4 different animals (bear, duck, deer,
 * turtle).
 *
 * References: - U.S. Fish and Wildlife Service (animal sample dataset):
 * http://digitalmedia.fws.gov/cdm/ - Tiny ImageNet Classification with CNN:
 * http://cs231n.stanford.edu/reports/2015/pdfs/leonyao_final.pdf
 *
 * CHALLENGE: Current setup gets low score results. Can you improve the scores?
 * Some approaches: - Add additional images to the dataset - Apply more
 * transforms to dataset - Increase epochs - Try different model configurations
 * - Tune by adjusting learning rate, updaters, activation & loss functions,
 * regularization, ...
 */

public class MagicCardClassificationThoemel {
	protected static final Logger log = LoggerFactory.getLogger(MagicCardClassificationThoemel.class);
	private SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH-mm-ss.SSS");

	protected static int height = 100;
	protected static int width = 100;
	protected static int channels = 3;
	protected static int numExamples = 80;
	protected static int batchSize = 20;

	protected static long seed = 42;
	protected static Random rng = new Random(seed);
	protected static int listenerFreq = 1;
	protected static int iterations = 1;
	protected static double splitTrainTest = 0.8;
	protected static boolean persistFinalModel = true;
	protected static boolean persistIntermediateModels = false;
	
	protected static int nCores = 4;
	protected static int numLabels = 13;
	protected static int epochs = 100;
	private static final Network algo = Network.DNN;
	
	private enum Network {
		DNN, LENET, ALEXNET
	}

	public void run(String[] args) throws Exception {

		log.info("Load data....");
		/**
		 * cd Data Setup -> organize and limit data file paths: - mainPath =
		 * path to image files - fileSplit = define basic dataset split with
		 * limits on format - pathFilter = define additional file load filter to
		 * limit size and balance batch content
		 **/
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		File mainPath = new File(System.getProperty("user.dir"), "src/main/resources/magic_cards/");

		String modelBasePath = FilenameUtils.concat(mainPath.getAbsolutePath(),
				"models/" + dateFormat.format(new Date()) + "/");
		new File(modelBasePath).mkdirs();
		File storedModel = new File(modelBasePath + "magic-model.zip");

		/**
		 * Data Setup -> train test split - inputSplit = define train and test
		 * split
		 **/

		InputSplit trainData = new FileSplit(new File(mainPath, "train"), NativeImageLoader.ALLOWED_FORMATS, rng);
		InputSplit testData = new FileSplit(new File(mainPath, "test"), NativeImageLoader.ALLOWED_FORMATS);

		/**
		 * Data Setup -> transformation - Transform = how to tranform images and
		 * generate large dataset to train on
		 **/
		DataNormalization scaler = new ImagePreProcessingScaler(0, 128);
		// ImageTransform resizeTransform = new ResizeImageTransform(height,
		// width);

		log.info("Build model....");

		
		MultiLayerNetwork network;
		ImageRecordReader recordReader;
		switch (algo) {
		case DNN:
			network = nullAchtFuenfzehn();
			recordReader = new FlatteningImageRecordReader(height, width, channels, labelMaker);
			break;
		case LENET:
			network = lenetModel();
			recordReader = new ImageRecordReader(height, width, channels, labelMaker);
			break;
		case ALEXNET:
			network = alexnetModel();
			recordReader = new ImageRecordReader(height, width, channels, labelMaker);
			break;
		default:
			throw new IllegalArgumentException("No algo chosen");
		}
		
		

		network.init();
		network.setListeners(new ScoreIterationListener(listenerFreq), new PerformanceListener(10));

		/**
		 * Data Setup -> define how to load data into net: - recordReader = the
		 * reader that loads and converts image data pass in inputSplit to
		 * initialize - dataIter = a generator that only loads one batch at a
		 * time into memory to save memory - trainIter = uses
		 * MultipleEpochsIterator to ensure model runs through the data for all
		 * epochs
		 **/
		DataSetIterator dataIter;
		MultipleEpochsIterator trainIter;

		// Train without transformations
		recordReader.initialize(trainData);
		dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);

		// scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);
		if (storedModel.exists()) {
			trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
			log.info("Load model from file ....");
			network = loadFromFile(storedModel);
		} else {
			trainIter = createMultipleEpochsIterator(storedModel, network, dataIter);
			log.info("Train model....");
			network.fit(trainIter);


			List<ImageTransform> transforms = new ArrayList<>();
//			for (int i : new int[] { 1, 2, 3, 5, 6, 7 }) {
//				transforms.add(new RotateImageTransform(new Float(i * 45)));
//			}

//			for (int i = 1; i < 24; i++) {
//				transforms.add(new RotateImageTransform(new Float(i * 15)));
//			}

//			ImageTransform flipImage1 = new FlipImageTransform(0);
//			ImageTransform flipImage2 = new FlipImageTransform(-1);
//			ImageTransform flipImage3 = new FlipImageTransform(1);
//			ImageTransform equalizeHist = new EqualizeHistTransform();
//			ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
//			// ImageTransform showImage = new ShowImageTransform("sali", 0);
//			// ImageTransform multi = new MultiImageTransform(colorTransform,
//			// showImage);
//			
//			transforms.addAll(Arrays.asList(new ImageTransform[] { flipImage1, flipImage2, flipImage3, equalizeHist, colorTransform }));

			// Train with transformations
			for (ImageTransform transform : transforms) {
				System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
				recordReader.initialize(trainData, transform);
				dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
				scaler.fit(dataIter);
				dataIter.setPreProcessor(scaler);
				trainIter = createMultipleEpochsIterator(storedModel, network, dataIter,
						transform.getClass().getSimpleName().toString());
				network.fit(trainIter);
			}

			if (persistFinalModel) {
				log.info("Save model....");
				ModelSerializer.writeModel(network, storedModel, true);
			}
		}

		log.info("Evaluate model....");
		recordReader.initialize(testData);
		dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
		// scaler.fit(dataIter);
		dataIter.setPreProcessor(scaler);
		Evaluation eval = network.evaluate(dataIter);
		log.info(eval.stats(true));
		log.info("\n" + eval.confusionToString());

		dataIter.reset();
		DecimalFormat df = new DecimalFormat("0.000");
		INDArray output = network.output(dataIter);
		for (int i = 0; i < output.size(0); i++) {
			URI uri = testData.locations()[i];
			String fileName = uri.getPath().substring(uri.getPath().lastIndexOf(File.separator) + 1);
			log.info("*****************");
			log.info(fileName);
			log.info("*****************");
			INDArray row = output.getRow(i);
			for (int j = 0; j < row.size(1); j++) {
				log.info("\t" + dataIter.getLabels().get(j) + ": " + df.format(row.getDouble(j)));
			}
		}

	}

	private MultipleEpochsIterator createMultipleEpochsIterator(File storedModel, MultiLayerNetwork network,
			DataSetIterator dataIter) {
		return createMultipleEpochsIterator(storedModel, network, dataIter, null);
	}

	private MultipleEpochsIterator createMultipleEpochsIterator(File storedModel, MultiLayerNetwork network,
			DataSetIterator dataIter, String transformationInfo) {
		MultipleEpochsIterator trainIter;
		if (persistIntermediateModels) {
			trainIter = new MultipleEpochsModelSavingIterator(epochs, dataIter, nCores, network,
					storedModel.getAbsolutePath(), Optional.ofNullable(transformationInfo));
		} else {
			trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores);
		}
		return trainIter;
	}

	private MultiLayerNetwork loadFromFile(File storedModel) throws IOException {
		return ModelSerializer.restoreMultiLayerNetwork(storedModel);

	}

	private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad,
			double bias) {
		return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
	}

	private ConvolutionLayer conv3x3(String name, int out, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 3, 3 }, new int[] { 1, 1 }, new int[] { 1, 1 }).name(name)
				.nOut(out).biasInit(bias).build();
	}

	private ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 5, 5 }, stride, pad).name(name).nOut(out).biasInit(bias)
				.build();
	}

	private SubsamplingLayer maxPool(String name, int[] kernel) {
		return new SubsamplingLayer.Builder(kernel, new int[] { 2, 2 }).name(name).build();
	}

	private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
		return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
	}

	private MultiLayerNetwork lenetModel() {
		/**
		 * Revisde Lenet Model approach developed by ramgo2 achieves slightly
		 * above random Reference:
		 * https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
		 **/
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
				.regularization(false).l2(0.005) // tried 0.0001, 0.0005
				.activation(Activation.RELU).learningRate(0.0001) // tried
																	// 0.00001,
																	// 0.00005,
																	// 0.000001
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.RMSPROP).momentum(0.9).list()
				.layer(0, convInit("cnn1", channels, 50, new int[] { 5, 5 }, new int[] { 1, 1 }, new int[] { 0, 0 }, 0))
				.layer(1, maxPool("maxpool1", new int[] { 2, 2 }))
				.layer(2, conv5x5("cnn2", 100, new int[] { 5, 5 }, new int[] { 1, 1 }, 0))
				.layer(3, maxPool("maxool2", new int[] { 2, 2 })).layer(4, new DenseLayer.Builder().nOut(500).build())
				.layer(5,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(numLabels)
								.activation(Activation.SOFTMAX).build())
				.backprop(true).pretrain(false).setInputType(InputType.convolutional(height, width, channels)).build();

		return new MultiLayerNetwork(conf);

	}

	private MultiLayerNetwork alexnetModel() {
		/**
		 * AlexNet model interpretation based on the original paper ImageNet
		 * Classification with Deep Convolutional Neural Networks and the
		 * imagenetExample code referenced.
		 * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
		 **/

		double nonZeroBias = 1;
		double dropOut = 0.5;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.0, 0.01)).activation(Activation.RELU)
				.updater(Updater.NESTEROVS).iterations(iterations)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize
																					// to
																					// prevent
																					// vanishing
																					// or
																					// exploding
																					// gradients
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).learningRate(1e-2)
				.biasLearningRate(1e-2 * 2).learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(0.1)
				.lrPolicySteps(100000).regularization(true).l2(5 * 1e-4).momentum(0.9).miniBatch(false).list()
				.layer(0,
						convInit("cnn1", channels, 96, new int[] { 11, 11 }, new int[] { 4, 4 }, new int[] { 3, 3 }, 0))
				.layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
				.layer(2, maxPool("maxpool1", new int[] { 3, 3 }))
				.layer(3, conv5x5("cnn2", 256, new int[] { 1, 1 }, new int[] { 2, 2 }, nonZeroBias))
				.layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
				.layer(5, maxPool("maxpool2", new int[] { 3, 3 })).layer(6, conv3x3("cnn3", 384, 0))
				.layer(7, conv3x3("cnn4", 384, nonZeroBias)).layer(8, conv3x3("cnn5", 256, nonZeroBias))
				.layer(9, maxPool("maxpool3", new int[] { 3, 3 }))
				.layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
				.layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
				.layer(12,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
								.nOut(numLabels).activation(Activation.SOFTMAX).build())
				.backprop(true).pretrain(false).setInputType(InputType.convolutional(height, width, channels)).build();

		return new MultiLayerNetwork(conf);

	}
	
	
	private MultiLayerNetwork nullAchtFuenfzehn() {
		int nIn = 30000;
		int hiddenNodes = 500;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(42)
				.iterations(iterations)
//				.activation(Activation.RELU)
				.activation(Activation.TANH)
				.weightInit(WeightInit.XAVIER)
				.learningRate(0.001)
                .regularization(true).l2(1e-4)
//				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//				.updater(Updater.NESTEROVS)
//				.momentum(0.9)
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(nIn)
						.nOut(hiddenNodes)
//						.weightInit(WeightInit.XAVIER)
//						.activation(Activation.RELU)
						.build())
                .layer(1, new DenseLayer.Builder().nIn(hiddenNodes).nOut(hiddenNodes).build())
				.layer(2, new OutputLayer.Builder()
						.nIn(hiddenNodes)
						.nOut(numLabels)
						.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation(Activation.SOFTMAX)
//						.weightInit(WeightInit.XAVIER)
						.build())
                .backprop(true).pretrain(false)
				.build();
		
		return new MultiLayerNetwork(conf);

	}


	public static void main(String[] args) throws Exception {
		new MagicCardClassificationThoemel().run(args);
	}

	private class MultipleEpochsModelSavingIterator extends MultipleEpochsIterator {

		private static final long serialVersionUID = 1L;
		private String rawFilePath;
		private Model model;
		private Optional<String> tag;

		public MultipleEpochsModelSavingIterator(int numEpochs, DataSetIterator iter, int queueSize,
				@NonNull Model model, @NonNull String filePath, Optional<String> tag) {
			super(numEpochs, iter, queueSize);
			this.model = model;
			this.rawFilePath = filePath;
			this.tag = tag;
		}

		@Override
		public void trackEpochs() {
			super.trackEpochs();
			saveModel();
		}

		private void saveModel() {
			String filenameWithAdditionalInfo = addInfoToFilePath(rawFilePath, "_epoch_" + epochs + "_");
			filenameWithAdditionalInfo = addInfoToFilePath(filenameWithAdditionalInfo,
					(tag.isPresent() ? tag.get() : "base") + "_");
			filenameWithAdditionalInfo = addInfoToFilePath(filenameWithAdditionalInfo, dateFormat.format(new Date()));
			try {
				ModelSerializer.writeModel(model, filenameWithAdditionalInfo, true);
				log.info("Model saved to file " + filenameWithAdditionalInfo + " for epoch " + epochs);
			} catch (IOException e) {
				log.warn("Could not write model to path " + filenameWithAdditionalInfo + ": " + e.getMessage());
			}

		}

		private String addInfoToFilePath(String filePath, String info) {
			int splitPoint = FilenameUtils.indexOfExtension(filePath);
			return filePath.substring(0, splitPoint) + info + filePath.substring(splitPoint);
		}
	}


	private class FlatteningImageRecordReader extends ImageRecordReader {
		
	    public FlatteningImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator) {
	        super(height, width, channels, labelGenerator);
	    }

		  @Override
		    public List<Writable> next() {
		        if (iter != null) {
		            List<Writable> ret;
		            File image = iter.next();
		            currentFile = image;

		            if (image.isDirectory())
		                return next();
		            try {
		                invokeListeners(image);
		                INDArray row = imageLoader.asMatrix(image);
		                INDArray flattenedRow = Nd4j.toFlattened('c', row.getRow(0));
		                ret = RecordConverter.toRecord(flattenedRow);
		                if (appendLabel)
		                    ret.add(new IntWritable(labels.indexOf(getLabel(image.getPath()))));
		            } catch (Exception e) {
		                throw new RuntimeException(e);
		            }
		            return ret;
		        } else if (record != null) {
		            hitImage = true;
		            invokeListeners(record);
		            return record;
		        }
		        throw new IllegalStateException("No more elements");
		    }

	}
	
}
