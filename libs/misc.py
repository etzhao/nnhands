def write_log(log_fp, epoch, idx, loss, loss_detailed, state, toprint=True):
     with open(log_fp, 'a') as f:
            loss_detailed = ", ".join(map(str, loss_detailed))
            towrite = "{}, {}, {}, {}\n".format(epoch, idx, loss, loss_detailed)
            f.write(towrite)
            if toprint:
                print('Epoch: {}, Iter: {}, {} Loss: {}'.format(epoch, idx, state, loss))

