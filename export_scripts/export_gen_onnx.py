import os
import json
import torch
import torch.onnx

class Exporter(object):
    def __init__(self, config):
        self.config = config

    def export(self):
        save_path_gen = "exports/{}_generator.onnx".format(self.config["version"])
        exports_dir = "exports"
        save_path_condition_labels = "{}/{}_condition-labels.json".format(exports_dir, self.config["version"])

        print("Loading trained model...")

        package = __import__(self.config["com_base"]+self.config["gScriptName"], fromlist=True)
        GClass  = getattr(package, 'Generator')
        
        Gen     = GClass(self.config["GConvDim"], self.config["GKS"], self.config["resNum"], len(self.config["selectedStyleDir"]))
        if self.config["cuda"] >=0:
            Gen = Gen.cuda()
        
        checkpoint = torch.load(self.config["ckp_name"])
        Gen.load_state_dict(checkpoint['g_model'])
        Gen.eval() # set to inference mode

        print("Exporting condition labels tensor as JSON...")

        batch_size  = self.config["batchSize"]
        n_class     = len(self.config["selectedStyleDir"])

        condition_labels = torch.ones((n_class, batch_size, 1)).long()
        for i in range(n_class):
            condition_labels[i,:,:] = condition_labels[i,:,:]*i

        if not os.path.exists(exports_dir):
            os.makedirs(exports_dir)
        with open(save_path_condition_labels, 'w') as f:
            json.dump(condition_labels.cpu().numpy().tolist(), f) # dump tensor as json before cuda alloc

        if self.config["cuda"] >=0:
            condition_labels = condition_labels.cuda()

        print("Exporting Generator as ONNX model...")

        dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True)
        if self.config["cuda"] >=0:
            dummy_input = dummy_input.cuda()

        dynamic_axes_mapping = {'input_img' : {2 : 'input_img_height', 3 : 'input_img_width'},
                                'output_img' : {2 : 'output_img_height', 3 : 'output_img_width'}}

        # Export the model
        torch.onnx.export(Gen,                                  # model being run
            (dummy_input, condition_labels[0, 0, :]),           # model input
            save_path_gen,                                      # where to save the model
            export_params = True,                               # store the trained parameter weights inside the model file
            opset_version = 11,                                 # the ONNX version to export the model to
            do_constant_folding = True,                         # whether to execute constant folding for optimization
            input_names = ['input_img', 'input_style'],         # the model's input names
            output_names = ['output_img'],                      # the model's output names
            dynamic_axes = dynamic_axes_mapping)
            
        print("Finished exporting Generator!")
