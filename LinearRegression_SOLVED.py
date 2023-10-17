
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr


def runcode(value1, value2):
    
    ti = int (value1)
    sr = float (value2)

    # generate random data-set
    #np.random.seed(0) # choose random seed (optional)
    x = np.random.rand(100, 1)
    y = 2 + 3 * x + np.random.rand(100, 1)

    J = 0 # initialize J, this can be deleted once J is defined in the loop
    w = np.matrix([np.random.rand(),np.random.rand()]) # slope and y-intercept
    a = sr # learning rate step size
    ite = ti # number of training iterations

    ## Write Linear Regression Code to Solve for w (slope and y-intercept) Here ##
    for p in range (ite):
        for i in range(len(x)):
            # Calculate w and J here
            x_vec = np.matrix([x[i][0],1]) # Setting up a vector for x (x_vec[j] corresponds to w[j])
            # h = (define h here) ## Hint: you may need to transpose x or w by adding .T to the end of the variable
            h = w * x_vec.T
            # w = (define w update iteration here)
            w = w - (a*(h-y[i]) * x_vec)
            # J = (loss equation here)
            J = 0.5 * (h - y[i])**2
        
        print('Loss:', J)

    ## if done correctly the line should be in line with the data points ##

    print('f = ', w[0,0],'x + ', w[0,1])
    equation = f"f = {w[0,0]}x + {w[0,1]}"
    # plot
    zp = w[0,1] + (w[0,0] * x)
    fig = plt.figure()
    plt.scatter(x,y,s=10)
    plt.plot(x, zp, linestyle='solid')
    plt.xlabel('x')
    plt.ylabel('y')
    return  fig, equation
    




with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")) as demo:
    gr.Markdown("Start typing below and then click **Run** to see the output.")
    with gr.Row():
    
        inp = gr.Number(label="# of Training Iterations")
        inp1 = gr.Number(label="Step Rate")
        fig = gr.Plot(type = "matplotlib")
        equation = gr.Textbox(label = "Loss")

    btn = gr.Button("Import")
    btn.click(fn=runcode, inputs=[inp, inp1], outputs = [fig, equation])

demo.launch()


