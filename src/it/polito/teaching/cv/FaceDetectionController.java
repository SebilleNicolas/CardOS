package it.polito.teaching.cv;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
//import org.opencv.videoio.VideoCapture;

import it.polito.elite.teaching.cv.utils.Utils;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the face detection/tracking.
 * 
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @version 1.1 (2015-11-10)
 * @since 1.0 (2014-01-10)
 * 		
 */
public class FaceDetectionController
{
	// FXML buttons
	@FXML
	private Button cameraButton;
	// the FXML area for showing the current frame
	@FXML
	private ImageView originalFrame;
	// checkboxes for enabling/disabling a classifier
	@FXML
	private CheckBox haarClassifier;
	@FXML
	private CheckBox lbpClassifier;
	
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that performs the video capture
	private VideoCapture capture;
	// a flag to change the button behavior
	private boolean cameraActive;
	
	// face cascade classifier
	private CascadeClassifier faceCascade;
	private CascadeClassifier eyesCascade;
	private CascadeClassifier mouthsCascade;
	private int absoluteFaceSize;
	
	/**
	 * Init the controller, at start time
	 */
	protected void init()
	{
		this.capture = new VideoCapture();
		this.faceCascade = new CascadeClassifier();
		this.eyesCascade = new CascadeClassifier();
		this.mouthsCascade = new CascadeClassifier();
		
		this.absoluteFaceSize = 0;
		
		// set a fixed width for the frame
		originalFrame.setFitWidth(600);
		// preserve image ratio
		originalFrame.setPreserveRatio(true);
		
		this.lbpClassifier.setSelected(false);
		this.cameraButton.setDisable(false);
		
	}
	
	/**
	 * The action triggered by pushing the button on the GUI
	 */
	@FXML
	protected void startCamera()
	{	
		if (!this.cameraActive)
		{
			// disable setting checkboxes
			this.haarClassifier.setDisable(true);
			this.lbpClassifier.setDisable(true);
			
			// start the video capture
			this.capture.open(0);
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(originalFrame, imageToShow);
					}
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				// update the button content
				this.cameraButton.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Failed to open the camera connection...");
			}
		}
		else
		{
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.cameraButton.setText("Start Camera");
			// enable classifiers checkboxes
			this.haarClassifier.setDisable(false);
			this.lbpClassifier.setDisable(false);
			
			// stop the timer
			this.stopAcquisition();
			System.exit(0);
		}
	}
	
	/**
	 * Get a frame from the opened video stream (if any)
	 * 
	 * @return the {@link Image} to show
	 */
	private Mat grabFrame()
	{
		Mat frame = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);
				
				// if the frame is not empty, process it
				if (!frame.empty())
				{
					// face detection
					this.detectAndDisplay(frame);
				}
				
			}
			catch (Exception e)
			{
				// log the (full) error
				System.err.println("Exception during the image elaboration: " + e);
			}
		}
		
		return frame;
	}
	
	/**
	 * Method for face detection and tracking
	 * 
	 * @param frame
	 *            it looks for faces in this frame
	 */
	private void detectAndDisplay(Mat frame)
	{
		MatOfRect faces = new MatOfRect();
		MatOfRect eyes = new MatOfRect();
		MatOfRect mouths = new MatOfRect();
		
		
		Mat grayFrame = new Mat();
		
		// convert the frame in gray scale
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		// equalize the frame histogram to improve the result
		Imgproc.equalizeHist(grayFrame, grayFrame);
		
		// compute minimum face size (20% of the frame height, in our case)
		if (this.absoluteFaceSize == 0)
		{
			int height = grayFrame.rows();
			if (Math.round(height * 0.2f) > 0)
			{
				this.absoluteFaceSize = Math.round(height * 0.2f);
			}
		}
		
		
//		// ------- FOURIER -------
//		// optimize the dimension of the loaded image
//				Mat padded = optimizeImageDim(grayFrame);
//				padded.convertTo(padded, CvType.CV_32F);
//				
//				List<Mat> planes = new ArrayList<>();
//				// prepare the image planes to obtain the complex image
//				planes.add(padded);
//				planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
//				// prepare a complex image for performing the dft
//				 Mat complexImage = new Mat();
//				Core.merge(planes, complexImage);
//				
//				// dft
//				Core.dft(complexImage, complexImage);
//				
//				// optimize the image resulting from the dft operation
//				Mat magnitude = createOptimizedMagnitude(complexImage);
////				
////				
////				IDFT -- Inverse fourier
//				Core.idft(complexImage, complexImage);
//				
//				Mat restoredImage = new Mat();
//				Core.split(complexImage, planes);
//				Core.normalize(planes.get(0), restoredImage, 0, 255, Core.NORM_MINMAX);
//				
//				// move back the Mat to 8 bit, in order to proper show the result
//				restoredImage.convertTo(restoredImage, CvType.CV_8U);
//				grayFrame=restoredImage;
//				
//		    		
		
		
//		-------------------------- Basic rectangle facial recognition --------------------------
//		// detect faces
//		this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//				new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());
//				
//		// each rectangle in faces is a face: draw them!
//		Rect[] facesArray = faces.toArray();
//		for (int i = 0; i < facesArray.length; i++)
//			Core.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
////			
//		// detect eyes
//		this.eyesCascade.detectMultiScale(grayFrame, eyes, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
//						new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());
//						
//				// each rectangle in eyes is a eye: draw them!
//				Rect[] eyesArray = faces.toArray();
//				for (int i = 0; i < eyesArray.length; i++)
//					Core.rectangle(frame, eyesArray[i].tl(), eyesArray[i].br(), new Scalar(255, 0, 0), 3);
//					
		
		
//		-------------------------- with Eyes and facial recognition --------------------------
		// detect faces
		this.faceCascade.detectMultiScale(frame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
				new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());
				
		// each rectangle in faces is a face: draw them!
		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++)
		{
			Core.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
//			

		        Mat faceROI = frame.submat(facesArray[i]);


		        eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0, new Size(30, 30), new Size());

		        Rect[] eyesArray = eyes.toArray();

		        for (int j = 0; j < eyesArray.length; j++) {
		            Point centre2 = new Point(facesArray[i].x + eyesArray[j].x + eyesArray[j].width * 0.5,
		                    facesArray[i].y + eyesArray[j].y + eyesArray[j].height * 0.5);
		            int radius = (int) Math.round((eyesArray[j].width + eyesArray[j].height) * 0.25);
		            Core.circle(frame, centre2, radius, new Scalar(255, 202, 121), 4, 8, 0);
		        }
		        
		      
		        
		    }
//		------------------------------------------------------------------------------
		
	}
	
	 /**
		 * Optimize the image dimensions
		 *
		 * @param image
		 *            the {@link Mat} to optimize
		 * @return the image whose dimensions have been optimized
		 */
		public Mat optimizeImageDim(Mat image)
		{
			// init
			Mat padded = new Mat();
			// get the optimal rows size for dft
			int addPixelRows = Core.getOptimalDFTSize(image.rows());
			// get the optimal cols size for dft
			int addPixelCols = Core.getOptimalDFTSize(image.cols());
			// apply the optimal cols and rows size to the image
			Imgproc.copyMakeBorder(image, padded, 0, addPixelRows - image.rows(), 0, addPixelCols - image.cols(),
					Imgproc.BORDER_CONSTANT, Scalar.all(0));

			return padded;
		}
		/**
		 * Optimize the magnitude of the complex image obtained from the DFT, to
		 * improve its visualization
		 * 
		 * @param complexImage
		 *            the complex image obtained from the DFT
		 * @return the optimized image
		 */
		public Mat createOptimizedMagnitude(Mat complexImage)
		{
			// init
			List<Mat> newPlanes = new ArrayList<>();
			Mat mag = new Mat();
			// split the comples image in two planes
			Core.split(complexImage, newPlanes);
			// compute the magnitude
			Core.magnitude(newPlanes.get(0), newPlanes.get(1), mag);
			
			// move to a logarithmic scale
			Core.add(Mat.ones(mag.size(), CvType.CV_32F), mag, mag);
			Core.log(mag, mag);
			// optionally reorder the 4 quadrants of the magnitude image
			shiftDFT(mag);
			// normalize the magnitude image for the visualization since both JavaFX
			// and OpenCV need images with value between 0 and 255
			// convert back to CV_8UC1
			mag.convertTo(mag, CvType.CV_8UC1);
			Core.normalize(mag, mag, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
			
			// you can also write on disk the resulting image...
			// Imgcodecs.imwrite("../magnitude.png", mag);
			
			return mag;
		}
		/**
		 * Reorder the 4 quadrants of the image representing the magnitude, after
		 * the DFT
		 * 
		 * @param image
		 *            the {@link Mat} object whose quadrants are to reorder
		 */
		public void shiftDFT(Mat image)
		{
			image = image.submat(new Rect(0, 0, image.cols() & -2, image.rows() & -2));
			int cx = image.cols() / 2;
			int cy = image.rows() / 2;
			
			Mat q0 = new Mat(image, new Rect(0, 0, cx, cy));
			Mat q1 = new Mat(image, new Rect(cx, 0, cx, cy));
			Mat q2 = new Mat(image, new Rect(0, cy, cx, cy));
			Mat q3 = new Mat(image, new Rect(cx, cy, cx, cy));
			
			Mat tmp = new Mat();
			q0.copyTo(tmp);
			q3.copyTo(q0);
			tmp.copyTo(q3);
			
			q1.copyTo(tmp);
			q2.copyTo(q1);
			tmp.copyTo(q2);
		}
	
	/**
	 * The action triggered by selecting the Haar Classifier checkbox. It loads
	 * the trained set to be used for frontal face detection.
	 */
	@FXML
	protected void haarSelected(Event event)
	{
		// check whether the lpb checkbox is selected and deselect it
		if (this.lbpClassifier.isSelected())
			this.lbpClassifier.setSelected(false);
			
		this.checkboxSelection("resources/haarcascades/haarcascade_frontalface_alt.xml");
		this.eyesCascade.load("resources/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
//		this.checkboxSelection("resources/haarcascades/haarcascade_frontalface_default.xml");
		
		//System.out.println("*haarcascade_frontalface_alt.xml* is selected !");
	}
	
	/**
	 * The action triggered by selecting the LBP Classifier checkbox. It loads
	 * the trained set to be used for frontal face detection.
	 */
	@FXML
	protected void lbpSelected(Event event)
	{
		// check whether the haar checkbox is selected and deselect it
		if (this.haarClassifier.isSelected())
			this.haarClassifier.setSelected(false);
	
		this.checkboxSelection("resources/lbpcascades/lbpcascade_frontalface.xml");
		//this.eyesCascade.load("resources/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
		//this.mouthsCascade.load("resources/haarcascades/haarcascade_mcs-mouth.xml");
		//this.eyesCascade.load("resources/lbpcascades/lbpcascade_profileface.xml");
		//System.out.println("*lbpcascade_frontalface.xml* is selected !");
	}
	
	/**
	 * Method for loading a classifier trained set from disk
	 * 
	 * @param classifierPath
	 *            the path on disk where a classifier trained set is located
	 */
	private void checkboxSelection(String classifierPath)
	{
		// load the classifier(s)
		this.faceCascade.load(classifierPath);
		
		
		// now the video capture can start
		this.cameraButton.setDisable(false);
	}
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition()
	{
		if (this.timer!=null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		
		if (this.capture.isOpened())
		{
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
	}
	
}
