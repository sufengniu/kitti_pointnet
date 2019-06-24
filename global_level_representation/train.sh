python train.py -f tensor_data  --cuda --max-epochs 20 --iterations 16 --batch-size 50

python encoder.py --model checkpoint/encoder_epoch_00000005.pth \
				  --input example/sample_0.p \
				  --output example/code --iterations 16  --cuda


python decoder.py --model checkpoint/decoder_epoch_00000005.pth \
				  --input example/code.npz  --iterations 16 --output example/reconstruction


python evaluation.py -f tensor_data  --cuda --max-epochs 20 --iterations 16 --batch-size 1


encoder: EncoderCell(
  (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (rnn1): ConvLSTMCell(64, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), hidden_kernel_size=(1, 1))
  (rnn2): ConvLSTMCell(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), hidden_kernel_size=(1, 1))
  (rnn3): ConvLSTMCell(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), hidden_kernel_size=(1, 1))
)
binarizer: Binarizer(
  (conv): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (sign): Sign()
)
decoder: DecoderCell(
  (conv1): Conv2d(32, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (rnn1): ConvLSTMCell(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), hidden_kernel_size=(1, 1))
  (rnn2): ConvLSTMCell(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), hidden_kernel_size=(1, 1))
  (rnn3): ConvLSTMCell(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), hidden_kernel_size=(3, 3))
  (rnn4): ConvLSTMCell(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), hidden_kernel_size=(3, 3))
  (conv2): Conv2d(32, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
)