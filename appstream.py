import numpy as np
import streamlit as st
from PIL import Image
import os
import torch
from torch.autograd import Variable
from torchvision import transforms
import model
import embed
MODEL_PATH= 'G:/python\DeepLearning/DeepLearningProjects/SkinDiseaseProject/notebooks/skidisprev1/model_best.pth.tar'




embedder = embed.ResNet152Embedder()
embedder= embedder.cuda()
model1 = model.FineTuneNet()
model1 = model1.cuda()


CLASS_NAME_TO_IX = {
    2: u'Acne and Rosacea',
    3 : u'Actinic Keratosis Basal Cell Carcinoma(Malignant Lesion)',
    11: u'Atopic Dermatitis Photos',
    17: u'Bullous Disease Photos',
    1: u'Cellulitis Impetigo and other Bacterial Infections',
    22: u'Eczema Photos',
    8: u'Exanthems and Drug Eruptions',
    6: u'Hair Loss Photos Alopecia and other Hair Diseases',
    12: u'Herpes HPV and other STDs Photos',
    4: u'Light Diseases and Disorders of Pigmentation',
    9: u'Lupus and other Connective Tissue diseases',
    16: u'Melanoma Skin Cancer Nevi and Moles',
    18: u'Nail Fungus and other Nail Disease',
    0: u'Poison Ivy Photos and other Contact Dermatitis',
    5: u'Psoriasis pictures Lichen Planus and related diseases',
    10: u'Scabies Lyme Disease and other Infestations and Bites',
    13:u'Seborrheic Keratoses and other Benign Tumors',
    10: u'Systemic Disease',
    21: u'Tinea Ringworm Candidiasis and other Fungal Infections',
    7: u'Urticaria Hives',
    15: u'Vascular Tumors',
    14: u'Vasculitis Photos',
    20: u'Warts Molluscum and other Viral Infections',
}



title= st.title(' Skin Disease Detection')

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    # img= img.convert('RGB')
    return img 

loader = transforms.Compose([
     transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def image_loader(img):
    """load image, returns cuda tensor"""
    image = loader(img).float()
    image = Variable(image)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()


def get_dataloader(embedding):
    embedding_size=2048

    data = torch.Tensor(embedding_size)
    embedding = torch.from_numpy(embedding)
    data= embedding.squeeze()
    return data

image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
if image_file is not None:

    # To See Details
    file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
    st.write(file_details)

    img = load_image(image_file)
    # print(len(img.getbands()), "Bands")
    st.image(img)

    with torch.no_grad():


        embedder.eval()
        inputs = image_loader(img)
        image_emb = embedder(inputs).cpu().numpy()

        data= get_dataloader(image_emb).cuda()


        checkpoint = torch.load(MODEL_PATH)
        model1.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(torch.load(MODEL_PATH, map_location= 'cuda'))
        model1.eval()

        
        
        
        print(data.shape, "Input shape")
        outputs = model1(data).type(torch.cuda.FloatTensor)
        # # output= model.predict(img)
        

        ref_outputs= torch.exp(outputs)
        # output_np = ref_outputs.cpu().data.numpy()
        # output_np = np.argmax(output_np)
        best_3= torch.topk(ref_outputs, 3)
        class_1= CLASS_NAME_TO_IX[int(best_3.indices[0].cpu().numpy())]
        class_2= CLASS_NAME_TO_IX[int(best_3.indices[1].cpu().numpy())]
        class_3= CLASS_NAME_TO_IX[int(best_3.indices[2].cpu().numpy())]
        st.write(f'Differential Diagnosis: \n \n {class_1}:{100*best_3.values[0]:.2f} % \n \n {class_2} : {100*best_3.values[1]:.2f}% \n \n {class_3} : {100*best_3.values[2]:.2f}%')
