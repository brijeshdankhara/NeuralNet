# Dankhara, Brijesh
# 1000-127-7373
# 2016-09-20
# Assignment_04

import numpy as np
import Tkinter as Tk
import matplotlib
import math as maa
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import colorsys

class ClDataSet:
    # This class encapsulates the data set
    # The data set includes input samples and targets
    def __init__(self):
        file_name = './data_4.csv'
        _data = data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
        #print full_data

        self.full_data = np.array(_data)
        self.close = np.array(_data[:,0])
        close_max = np.amax(self.close,0)
        close_min = np.amin(self.close,0)
        col_diff = close_max - close_min
        self.close = (self.close - close_min)/(close_max-close_min)
        self.volume = np.array(_data[:,1])
        vol_max = np.amax(self.volume,0)
        vol_min = np.amin(self.volume,0)
        vol_diff = vol_max - vol_min
        self.volume = (self.volume - vol_min)/(vol_max-vol_min)

        for i in range(len(_data)):
            d = _data[i]
            clo = d[0]
            vol = d[1]
            clo = (clo - close_min) / col_diff
            vol = (vol - vol_min) / vol_diff
            d[0] = clo
            d[1] = vol
            _data[i] = d
        self.full_data = np.array(_data)
        print "a"

nn_experiment_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 2736,  # number of inputs to the network
    "learning_rate": 0.1,  # learning rate
    "sample_size": 10,  # momentum
    "batch_size": 200,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 2, "activation_function": "hardlimit"}],  # list of dictionaries
    "data_set": ClDataSet(),
    'number_of_delayed_elements': 3,
    'number_of_iteration': 3,
    'number_of_neurons':2
}


class ClNNExperiment:
    """
    This class presents an experimental setup for a single layer Perceptron
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings={}):
        self.error_vals = []
        self.batch_vals = 0
        self.no_batches = 0
        self.__dict__.update(nn_experiment_default_settings)
        self.__dict__.update(settings)
        self.input_no = self.number_of_inputs
        # Set up the neural network
        settings = {"min_initial_weights": self.min_initial_weights,  # minimum initial weight
                    "max_initial_weights": self.max_initial_weights,  # maximum initial weight
                    "number_of_inputs": self.number_of_delayed_elements,  # number of inputs to the network
                    "learning_rate": self.learning_rate,  # learning rate
                    "layers_specification": self.layers_specification
                    }
        self.neural_network = ClNeuralNetwork(self, settings)
        self.neural_network.layers[-1].number_of_neurons = 2

    def adjust_weights(self):

        # Setting the samples
        all_data = self.data_set.full_data
        self.add_RMS = []
        self.Abs_Max = []
        flo_size =float(self.sample_size)/100
        floor_val = maa.floor(flo_size * self.input_no)
        print floor_val
        dat = []
        for y in range(int(floor_val)):
            dat.append(np.array(all_data[y,:]))
        self.samples = np.array(dat)

        # Setting batch size
        real_data = []
        b_size = self.batch_size
        self.no_batches = maa.floor(floor_val/b_size)
        print self.no_batches


        out_pair = []
        # Setting number of iterations
        itera = self.number_of_iteration
        for x in range(len(self.samples)):
            self.error_vals = []
            if (x == (self.samples.shape[0] - 1)):
                real_data.append(self.samples[x, :])
                real_data = np.array(real_data)
                delayy = self.number_of_delayed_elements
                for k in range(self.number_of_iteration):
                    for i in range(len(real_data) - delayy):
                        target_index = int((i + delayy))
                        target_now = real_data[target_index, :]
                        actual_sample = []
                        for j in range(delayy):
                            sample_index = int(i + delayy - j)
                            actual_sample.append(real_data[sample_index, :])
                        act_sample = np.array(actual_sample);


                        self.neural_network.adjust_weights(act_sample.ravel(),
                                                           np.array(target_now) - self.neural_network.calculate_output(
                                                               act_sample.ravel()))

                        out_pair.append(np.array(target_now) - self.neural_network.calculate_output(
                            act_sample.ravel()))
                abs_m = np.array(out_pair)
                abs_m = np.amax(map(abs,abs_m))
                for aa in range(len(out_pair)):
                    error = (np.sqrt(np.mean(out_pair[aa] * out_pair[aa])))
                self.add_RMS.append(error)
                self.Abs_Max.append(abs_m)
                real_data = []
                out_pair = []
                print x
                break


            elif  (len(real_data) < b_size):
                real_data.append(self.samples[x,:])
                #print x


            elif (len(real_data) == b_size):
                real_data.append(self.samples[x, :])
                real_data = np.array(real_data)
                delayy = self.number_of_delayed_elements
                for k in range(self.number_of_iteration):
                    for i in range(len(real_data)-delayy):
                        target_index = int((i + delayy))
                        target_now = real_data[target_index,:]
                        actual_sample = []
                        for j in range(delayy):
                            sample_index = int(i+delayy-j)
                            actual_sample.append(real_data[sample_index,:])
                        act_sample = np.array(actual_sample)
                        self.neural_network.adjust_weights(act_sample.ravel(),
                                                       np.array(target_now) - self.neural_network.calculate_output(
                                                           act_sample.ravel()))
                        out_pair.append(np.array(target_now) - self.neural_network.calculate_output(
                                                           act_sample.ravel()))
                abs_m = np.array(out_pair)
                abs_m = np.amax(map(abs,abs_m))
                for aa in range(len(out_pair)):
                    error = (np.sqrt(np.mean(out_pair[aa] * out_pair[aa])))
                self.add_RMS.append(error)
                self.Abs_Max.append(abs_m)
                #print x
                real_data = []
                out_pair = []



class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    Farhad Kamangar 2016_09_02
    """

    def __init__(self,master, nn_experiment):
        self.master = master
        self.nn_experiment = nn_experiment
        self.xmin = -1
        self.xmax = 30
        self.ymin = -0.1
        self.ymax = 0.5
        self.master.update()

        # TODO: set default values of all sliders in experiment class
        self.number_of_delayed_elements = self.nn_experiment.number_of_delayed_elements
        self.learning_rate = self.nn_experiment.learning_rate
        # self.adjusted_learning_rate = self.learning_rate / self.number_of_samples_in_each_class
        self.sample_size = self.nn_experiment.sample_size
        self.batch_size = self.nn_experiment.batch_size
        self.number_of_iteration = self.nn_experiment.number_of_iteration
        self.step_size = 1
        self.current_sample_loss = 0
        self.sample_points = []
        self.target = []
        self.sample_colors = []
        self.weights = np.array([])
        self.class_ids = np.array([])
        self.output = np.array([])
        self.desired_target_vectors = np.array([])
        self.xx = np.array([])
        self.yy = np.array([])
        self.loss_type = ""
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        plt.title("LMS Implementation")
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Create sliders frame
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        # Set up the sliders
        ivar = Tk.IntVar()

        # Learning rate slider
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.001, to_=1, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # Number of delayed element slider
        self.number_of_delayed_elements_slider_label = Tk.Label(self.sliders_frame, text="Number of Delayed Elements")
        self.number_of_delayed_elements_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_delayed_elements_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                 from_=1, to_=1000, bg="#DDDDDD",resolution=1,
                                                 activebackground="#FF0000",
                                                 highlightcolor="#00FFFF", width=10)
        self.number_of_delayed_elements_slider.set(self.number_of_delayed_elements)
        self.number_of_delayed_elements_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_delayed_elements_callback())
        self.number_of_delayed_elements_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # sample size slider
        self.sample_size_slider_label = Tk.Label(self.sliders_frame, text="Sample Size")
        self.sample_size_slider_label.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.sample_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.sample_size_slider.set(self.sample_size)
        self.sample_size_slider.bind("<ButtonRelease-1>", lambda event: self.sample_size_slider_callback())
        self.sample_size_slider.grid(row=2, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # batch size slider
        self.batch_size_slider_label = Tk.Label(self.sliders_frame, text="Batch Size")
        self.batch_size_slider_label.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.batch_size_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=100, to_=500, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.batch_size_slider.set(self.batch_size)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=3, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # Number of iteration slider
        self.number_of_iteration_slider_label = Tk.Label(self.sliders_frame, text="No. of Iterations")
        self.number_of_iteration_slider_label.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_iteration_elements_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=1, to_=10, resolution=1, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.number_of_iteration_elements_slider.set(self.learning_rate)
        self.number_of_iteration_elements_slider.bind("<ButtonRelease-1>", lambda event: self.number_of_iteration_slider_callback())
        self.number_of_iteration_elements_slider.grid(row=4, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # set weights to zero btn
        self.set_weight_to_zero_bottun = Tk.Button(self.buttons_frame,
                                                   text="Set weight to zero",
                                                   bg="yellow", fg="red",
                                                   command=lambda: self.set_weight_to_zero_bottun_callback())
        self.set_weight_to_zero_bottun.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # adjust weights btn
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)


    # Plot Error
    def refresh_display_for_graph(self):
        errList = self.nn_experiment.add_RMS
        batches = self.nn_experiment.no_batches
        AM = self.nn_experiment.Abs_Max
        errBatches = []
        print errList
        for i in range(int(batches + 1)):
            if i is not batches:
                errBatches.append(i);
            else:
                break
        plt.plot(errBatches,errList)
        plt.plot(errBatches, AM, '*')
        self.canvas.draw()

    def refresh_display(self):
        print self.batch_size
        print self.sample_size
        print self.number_of_iteration
        print self.number_of_delayed_elements

    def learning_rate_slider_callback(self):
        self.learning_rate = self.learning_rate_slider.get()
        self.nn_experiment.learning_rate = self.learning_rate
        self.nn_experiment.neural_network.learning_rate = self.learning_rate
        self.refresh_display()

    def number_of_delayed_elements_callback(self):
        self.number_of_delayed_elements = self.number_of_delayed_elements_slider.get()
        self.nn_experiment.number_of_delayed_elements = self.number_of_delayed_elements
        self.refresh_display()

    def number_of_iteration_slider_callback(self):
        self.number_of_iteration = self.number_of_iteration_elements_slider.get()
        self.nn_experiment.number_of_iteration = self.number_of_iteration
        self.refresh_display()

    def set_weight_to_zero_bottun_callback(self):
        self.layer.weight = []
        # TODO: set weights to zero here

    def adjust_weights_button_callback(self):
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')
        #for k in range(10):
        self.randomize_weights()
        self.nn_experiment.adjust_weights()
        self.refresh_display_for_graph()
        self.refresh_display()
        #self.adjust_weights_button.config(text=temp_text)
        #self.adjust_weights_button.update_idletasks()

    def sample_size_slider_callback(self):
        self.sample_size = self.sample_size_slider.get()
        self.nn_experiment.sample_size = self.sample_size
        self.refresh_display()

    def batch_size_slider_callback(self):
        self.batch_size = self.batch_size_slider.get()
        self.nn_experiment.batch_size = self.batch_size
        self.refresh_display()

    def randomize_weights(self, min_initial_weights=-0.1, max_initial_weights=0.1):
        if min_initial_weights == None:
            min_initial_weights = self.min_initial_weights
        if max_initial_weights == None:
            max_initial_weights = self.max_initial_weights
        self.weights = np.random.uniform(min_initial_weights, max_initial_weights,
                                         (2, self.number_of_delayed_elements * 2 + 1))

neural_network_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs": 2,  # number of inputs to the network
    "learning_rate": 0.001,  # learning rate
    "momentum": 0.1,  # momentum
    "batch_size": 0,  # 0 := entire trainingset as a batch
    "layers_specification": [{"number_of_neurons": 2,
                              "activation_function": "linear"}]  # list of dictionaries
}

class ClNeuralNetwork:
    """
    This class presents a multi layer neural network
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, experiment, settings={}):
        self.__dict__.update(neural_network_default_settings)
        self.__dict__.update(settings)
        # create nn
        self.experiment = experiment
        self.layers = []
        for layer_index, layer in enumerate(self.layers_specification):
            if layer_index == 0:
                layer['number_of_inputs_to_layer'] = self.number_of_inputs
            else:
                layer['number_of_inputs_to_layer'] = self.layers[layer_index - 1].number_of_neurons
            self.layers.append(ClSingleLayer(layer))

    def randomize_weights(self, inputs ,min=-0.1, max=0.1):
        # randomize weights for all the connections in the network
        for layer in self.layers:
            layer.randomize_weights(self.min_initial_weights, self.max_initial_weights, inputs)

    def display_network_parameters(self, display_layers=True, display_weights=True):
        for layer_index, layer in enumerate(self.layers):
            print "\n--------------------------------------------", \
                "\nLayer #: ", layer_index, \
                "\nNumber of Nodes : ", layer.number_of_neurons, \
                "\nNumber of inputs : ", self.layers[layer_index].number_of_inputs_to_layer, \
                "\nActivation Function : ", layer.activation_function, \
                "\nWeights : ", layer.weights

    def calculate_output(self, input_values):
        # Calculate the output of the network, given the input signals
        for layer_index, layer in enumerate(self.layers):
            #print layer.weights
            if layer_index == 0:
                output = layer.calculate_output(input_values)
            else:
                output = layer.calculate_output(output)
        self.output = output
        return self.output

    def adjust_weights(self, input_samples, error):
        num_of_samples = len(input_samples.T)
        num_of_input = len(input_samples)
        input_samples = np.reshape(input_samples,(num_of_input,1))
        lr = self.learning_rate;
        for layer_index, layer in enumerate(self.layers):
            update2 = np.ones((num_of_input + 1, 1))
            update2[:-1, :] = np.array(input_samples)
            layer.weights = layer.weights + (2*lr*error * update2).T
        #print layer.weights

        return layer.weights

single_layer_default_settings = {
    # Optional settings
    "min_initial_weights": -0.1,  # minimum initial weight
    "max_initial_weights": 0.1,  # maximum initial weight
    "number_of_inputs_to_layer": 3,  # number of input signals
    "number_of_neurons": 2,  # number of neurons in the layer
    "activation_function": "linear"  # default activation function
}

class ClSingleLayer:
    """
    This class presents a single layer of neurons
    Farhad Kamangar 2016_09_04
    """

    def __init__(self, settings):
        self.__dict__.update(single_layer_default_settings)
        self.__dict__.update(settings)
        self.randomize_weights(self.number_of_inputs_to_layer)

    def randomize_weights(self,inputs, min_initial_weights=None, max_initial_weights=None):
        if min_initial_weights == None:
            min_initial_weights = self.min_initial_weights
        if max_initial_weights == None:
            max_initial_weights = self.max_initial_weights
        self.weights = np.random.uniform(min_initial_weights, max_initial_weights,
                                         (self.number_of_neurons, inputs*2 + 1))

    def calculate_output(self, input_values):
        # Calculate the output of the layer, given the input signals
        # NOTE: Input is assumed to be a column vector. If the input
        # is given as a matrix, then each column of the input matrix is assumed to be a sample
        # Farhad Kamangar Sept. 4, 2016
        if len(input_values.shape) == 1:
            net = self.weights.dot(np.append(input_values, 1))
            #print net
        else:
            net = self.weights.dot(np.vstack([input_values, np.ones((1, input_values.shape[1]), float)]))
        if self.activation_function == 'linear':
            self.output = net
        if self.activation_function == 'sigmoid':
            self.output = sigmoid(net)
        if self.activation_function == 'hardlimit':
            np.putmask(net, net > 0, 1)
            np.putmask(net, net <= 0, 0)
            self.output = net
        return self.output


if __name__ == "__main__":
    nn_experiment_settings = {
        # Optional settings
        "min_initial_weights": -0.1,  # minimum initial weight
        "max_initial_weights": 0.1,  # maximum initial weight
        "number_of_inputs": 2736,  # number of inputs to the network
        "learning_rate": 0.1,  # learning rate
        "sample_size": 10,  # momentum
        "batch_size": 200,  # 0 := entire trainingset as a batch
        "layers_specification": [{"number_of_neurons": 2, "activation_function": "linear"}],  # list of dictionaries
        'number_of_delayed_elements': 3,
        'number_of_iteration': 3
    }
    ob_nn_experiment = ClNNExperiment(nn_experiment_settings)
    np.random.seed(1)
    main_frame = Tk.Tk()
    main_frame.title("LMS")
    main_frame.geometry('640x480')
    ob_nn_gui_2d = ClNNGui2d(main_frame,ob_nn_experiment)
    main_frame.mainloop()