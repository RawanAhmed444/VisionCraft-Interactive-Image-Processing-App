# File: processor_factory.py

from  classes.edge_detector import EdgeDetector
# from  classes.active_contour_processor import ActiveContourProcessor
from  classes.noise_processor import NoiseProcessor
# from  classes.filter_processor import FilterProcessor
from  classes.frequency_processor import FrequencyProcessor
from  classes.histogram_processor import HistogramProcessor
# from  classes.hybrid_processor import HybridProcessor
from  classes.image_processor import ImageProcessor
from  classes.thresholding_processor import ThresholdingProcessor
from classes.active_contour_processor import ActiveContourProcessor

class ProcessorFactory:
    """
    A simple factory class to create different types of processor objects.
    """

    @staticmethod
    def create_processor(processor_type, *args, **kwargs):
        """
        Creates and returns an instance of a processor class based on processor_type.
        
        :param processor_type: A string that identifies which processor to create.
        :param args, kwargs: Additional arguments passed to the processor constructor.
        :return: An instance of the requested processor class.
        """
        processor_map = {
            "edge_detector": EdgeDetector,
            "noise": NoiseProcessor,
            "frequency": FrequencyProcessor,
            "histogram": HistogramProcessor,
            "thresholding": ThresholdingProcessor,
            "image": ImageProcessor,
            "active_contour": ActiveContourProcessor,
            
        }

        if processor_type not in processor_map:
            raise ValueError(f"Unknown processor type '{processor_type}'. Available types: {list(processor_map.keys())}")

        # Retrieve the class from the map
        processor_class = processor_map[processor_type]

        # Instantiate the processor with any provided args/kwargs
        return processor_class(*args, **kwargs)
