class Config:
    def __init__(self,
                use_reconst_loss,
                use_labels,
                num_classes=10,
                beta1=0.5,
                beta2=0.999,
                g_conv_dim=64,
                d_conv_dim=64,
                train_iters=40000,
                batch_size=64,
                lr=.0002,
                log_step=10,
                sample_step=500,
                sample_path='./samples',
                model_path='./models',
                image_size=32):
        self.use_reconst_loss = use_reconst_loss
        self.use_labels = use_labels
        self.num_classes = num_classes
        self.beta1 = beta1
        self.beta2 = beta2
        self.g_conv_dim = g_conv_dim
        self.d_conv_dim = d_conv_dim
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.lr = lr
        self.log_step = log_step
        self.sample_step = sample_step
        self.sample_path = sample_path
        self.model_path = model_path
        self.image_size = image_size
    #
#