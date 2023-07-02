# import necessary modules
import streamlit as st
import random

# import app modules (controller, service, and view)
from controller.df_dataset import dataset_df
from controller.mfcc import create_mfcc, create_resized_mfcc
from controller.cnn import train, test, predict
from service.split_data import make_train_test_split
from view.plot import distribution, show_random_plot, confusion_matrix, show_mfcc, plot_history
from view.config import get_config

# prepare current page config, set the page's title
get_config("Machine Modeling")

# prepare all needed session states to make interactive interaction
if "preprocessing" not in st.session_state: st.session_state["preprocessing"] = False
if "train" not in st.session_state: st.session_state["train"] = False
if "test" not in st.session_state: st.session_state["test"] = False
if "model" not in st.session_state: st.session_state["model"] = False
if "x_mean" not in st.session_state: st.session_state["x_mean"] = False
if "x_std" not in st.session_state: st.session_state["x_std"] = False

# start the machine modeling page
# used for preprocessing dataset, training and testing model
def start():
    # show the page title
    st.write("<h1 style='text-align: center;'>Machine Modeling</h1>", unsafe_allow_html = True)

    # show training dataset section
    st.write("### Training Dataset")
    # set imported dataset to global
    global df
    # load dataset
    df = dataset_df('dataset')
    # show dataset
    st.dataframe(df, 500)

    # show message and button for next interaction (preprocessing)
    st.write("Dataset have been loaded, let's preprocess them!")
    # set the preprocessing state to true if user click the button
    if st.button("Preprocessing", type = "primary"):
        st.session_state["preprocessing"] = True
    
# preprocessing function to start preprocessing
def preprocessing():
    # show sample distribution
    st.write("### Sample Distribution")
    distribution(df)

    # random one pick sample
    st.write("### Random Pick Sample")
    # split dataset to happy and sad based on their label, reset the dataframe index
    happy_df = df.loc[df["Label"] == "happy"].reset_index(drop = True)
    sad_df = df.loc[df["Label"] == "sad"].reset_index(drop = True)

    # plot 1 happy and 1 sad dataset
    st.write("#### Happy")
    show_random_plot(happy_df)
    st.write("#### Sad")
    show_random_plot(sad_df)

    # create the mfcc from each audio path
    mfccs = create_mfcc(df)
    # set the new_mfccs as global to be used in other function
    global new_mfccs
    # get the new mfccs (with the equal column number) and get the average column number of old mfccs
    new_mfccs, average_cols = create_resized_mfcc(mfccs)

    # show the average mfccs column (old mfccs)
    st.write("### Average MFCCs Column")
    st.metric("Average", average_cols)

    # pick one random index to show the difference of old and new mfcc
    index = random.randint(0, df.shape[0])
    # show picked mfcc, before and after equalization
    st.write("### MFCC Comparison")
    show_mfcc(df.Path[index], new_mfccs[index])

    # x_tr => training data, y_tr => training label
    # x_va => validation data, y_va => validation label
    # x_te => testing data, y_te => testing label
    global x_tr, y_tr, x_va, y_va, x_te, y_te

    # split the dataset into training, validation, and testing
    x_tr, y_tr, x_va, y_va, x_te, y_te = make_train_test_split(new_mfccs, df)

    # show the shape of mfccs distribution (training, validation, testing)
    st.write("### MFCCs Distribution")
    col1, col2, col3 = st.columns(3)
    col1.metric("Train", str(x_tr.shape))
    col2.metric("Validation", str(x_va.shape))
    col3.metric("Test", str(x_te.shape))

# start training process
def start_train():
    # show the training process section
    st.write("### Training Process")
    # start training
    model, history = train(x_tr, y_tr, x_va, y_va)
    # store the trained model in session state to be used in another function and page
    st.session_state["model"] = model
    # plot the training history (loss, validaton loss, accuracy, and validation accuracy)
    plot_history(history)

# start testing process
def start_test():
    # show the testing process section
    st.write("### Testing Process")
    # get the trained model from session state
    trained_model = st.session_state["model"]
    # test the model using testing data (return the loss and accuracy)
    loss, accuracy = test(trained_model, x_te, y_te)
    # get the prediction, precision, recall, and f1 score
    prediction, precision, recall, f1 = predict(trained_model, x_te, y_te)

    # display the testing metrics (loss, accuracy, precision, recall, and f1)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Loss", "{:.4f}".format(loss))
    col2.metric("Accuracy", "{:.4f}".format(accuracy))
    col3.metric("Precision", "{:.4f}".format(precision))
    col4.metric("Recall", "{:.4}".format(recall))
    col5.metric("F1", "{:.4}".format(f1))

    # plot the confusion matrix
    st.write("### Confusion Matrix")
    confusion_matrix(y_te, prediction)

# call start() function at the first time page load
if __name__ == "__main__":
    start()

# if the preprocessing state is true
if st.session_state["preprocessing"]:
    # start preprocessing
    preprocessing()
    # show message and button for next interaction (testing)
    st.write("The dataset have been preprocessed, let's train them using CNN!")
    # if training button pressed, change train state to true
    if st.button("Training", type = "primary"):
        st.session_state["train"] = True

# if the train state is true
if st.session_state["train"]:
    # start training process
    start_train()
    # show message and button for next interaction (testing)
    st.write("Training model fisinh, let's check the accuration with testing dataset!")
    # if testing button pressed, change test state to true
    if st.button("Testing", type = "primary"):
        st.session_state["test"] = True

# if the test state is true
if st.session_state["test"]:
    # start testing process
    start_test()

# code by @tudemaha