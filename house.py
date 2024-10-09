import streamlit as st 
import pickle
from PIL import Image

def main():
    st.title(":rainbow[PARIS HOUSE CLASSIFICATION]")
    image=Image.open('classimage.jpg')
    st.image(image,width=600)

    squareMeters=st.text_input(':red[squareMeters]')

    numberOfRooms=st.text_input(':red[numberOfRooms]')

    floors=st.text_input(':red[floors]')

    cityCode=st.text_input(':red[cityCode]')

    basement=st.text_input(':red[basement]')

    attic=st.text_input(':red[attic]')
    
    garage=st.text_input(':red[garage]')

    price=st.text_input(':red[price]')

    made=st.text_input(':red[made]')

    hasGuestRoom=st.text_input(':red[hasGuestRoom]')


    opt = ['0', '1']
    hasYard=st.radio(':green[hasYard]',opt)

    hasPool=st.radio(':green[hasPool]',opt)

    isNewBuilt=st.radio(':green[isNewBuilt]',opt)
    
    hasStormProtector=st.radio(':green[hasStormProtector]',opt)
    
    hasStorageRoom=st.radio(':green[hasStorageRoom]',opt)

    

    opt2=['1','2','3','4','5','6','7','8','9','10']
    cityPartRange=st.selectbox(':blue[cityPartRange]',opt2)

    opt3=['1','2','3','4','5','6','7','8','9','10']
    numPrevOwners=st.selectbox(':blue[numPrevOwners]',opt3)
    
    

    features=[squareMeters, numberOfRooms,hasYard,hasPool,floors,cityCode,cityPartRange,numPrevOwners,made,isNewBuilt,hasStormProtector,basement,attic,garage,hasStorageRoom,hasGuestRoom,price]  

    model=pickle.load(open('naivebayes.sav','rb'))
    standard=pickle.load(open('standard.sav','rb'))

    pred=st.button('PREDICT')

    if pred:
        prediction=model.predict(standard.transform([features]))

        if prediction==0:
            st.write('IT IS A BASIC HOUSE')

        else:
            st.write('IT IS A LUXURIOUS HOUSE')
  
main()
