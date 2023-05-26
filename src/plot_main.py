import pickle
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from options import args_parser
from utils import exp_details


if __name__ == '__main__':

    args = args_parser()
    exp_details(args)

    # fedavglayer = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_p{}.pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs, args.unequal, args.p)
    # fedavg = '../save/objects/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs, args.unequal)
    # fedprox = '../save/objects/fedprox_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs, args.unequal)
    # # fedproxlayer = '../save/objects/fedproxlayer_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_p{}.pkl'.\
    # #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    # #            args.local_ep, args.local_bs, args.unequal, args.p)
    #
    # with open(fedavglayer, 'rb') as f:
    #     train_loss1, train_accuracy1, global_loss1, global_acc1 = pickle.load(f)
    # with open(fedavg, 'rb') as f:
    #     train_loss2, train_accuracy2, global_loss2, global_acc2 = pickle.load(f)
    # # with open(fedproxlayer, 'rb') as f:
    # #     train_loss3, train_accuracy3, global_loss3, global_acc3 = pickle.load(f)
    # with open(fedprox, 'rb') as f:
    #     train_loss4, train_accuracy4, global_loss4, global_acc4 = pickle.load(f)


    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss1)), train_loss1, 'r', label='FedAvg-layer_aggre')
    # plt.plot(range(len(train_loss2)), train_loss2, 'k', label='FedAvg')
    # # plt.plot(range(len(train_loss3)), train_loss3, 'm', label='FedProx-layer_aggre')
    # plt.plot(range(len(train_loss4)), train_loss4, 'g', label='FedProx')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.legend()
    # plt.savefig('../save/all_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs, args.unequal))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy1)), train_accuracy1, 'r', label='FedAvg-layer_aggre')
    # plt.plot(range(len(train_accuracy2)), train_accuracy2, 'k', label='FedAvg')
    # # plt.plot(range(len(train_accuracy3)), train_accuracy3, 'm', label='FedProx-layer_aggre')
    # plt.plot(range(len(train_accuracy4)), train_accuracy4, 'g', label='FedProx')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.legend()
    # plt.savefig('../save/all_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs, args.unequal))
    #
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Testing Loss vs Communication rounds')
    # plt.plot(range(len(global_loss1)), global_loss1, 'r', label='FedAvg-layer_aggre')
    # plt.plot(range(len(global_loss2)), global_loss2, 'k', label='FedAvg')
    # # plt.plot(range(len(global_loss3)), global_loss3, 'm', label='FedProx-layer_aggre')
    # plt.plot(range(len(global_loss4)), global_loss4, 'g', label='FedProx')
    # plt.ylabel('Testing loss')
    # plt.xlabel('Communication Rounds')
    # plt.legend()
    # plt.savefig('../save/alltest_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs, args.unequal))

    # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('MNIST IID E=10 p=2')
    # plt.plot(range(len(global_acc1)), global_acc1, 'r', label='FedAvgLA')
    # plt.plot(range(len(global_acc2)), global_acc2, 'k', label='FedAvg')
    # # plt.plot(range(len(global_acc3)), global_acc3, 'm', label='FedProx-layer_aggre')
    # plt.plot(range(len(global_acc4)), global_acc4, 'g', label='FedProx')
    # ax = plt.gca()
    # x_space = MultipleLocator(2)
    # ax.xaxis.set_major_locator(x_space)
    # ay = plt.gca()
    # y_space = MultipleLocator(0.05)
    # ay.yaxis.set_major_locator(y_space)
    # plt.xlim([0, 50])
    # plt.ylim([0, 0.45])
    # plt.ylabel('Testing Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.legend()
    # plt.grid()
    # plt.savefig('../save/alltest_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs, args.unequal))


    fedavg = '../save/objects/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.unequal)
    fedavglayer1 = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_p1.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.unequal)
    fedavglayer2 = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_p2.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.unequal)
    fedavglayer3 = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_p3.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.unequal)
    fedavglayer4 = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_p4.pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.unequal)

    with open(fedavg, 'rb') as f:
        train_loss1, train_accuracy1, global_loss1, global_acc1 = pickle.load(f)
    with open(fedavglayer1, 'rb') as f:
        train_loss2, train_accuracy2, global_loss2, global_acc2 = pickle.load(f)
    with open(fedavglayer2, 'rb') as f:
        train_loss3, train_accuracy3, global_loss3, global_acc3 = pickle.load(f)
    with open(fedavglayer3, 'rb') as f:
        train_loss4, train_accuracy4, global_loss4, global_acc4 = pickle.load(f)
    with open(fedavglayer4, 'rb') as f:
        train_loss5, train_accuracy5, global_loss5, global_acc5 = pickle.load(f)

    # # Plot Loss curve
    # plt.figure()
    # plt.title('MNIST IID E=3')
    # plt.plot(range(len(train_accuracy1)), train_accuracy1, 'r', label='FedAvg-layer_p=0')
    # plt.plot(range(len(train_accuracy2)), train_accuracy2, 'k', label='FedAvg-layer_p=1')
    # plt.plot(range(len(train_accuracy3)), train_accuracy3, 'b', label='FedAvg-layer_p=2')
    # plt.plot(range(len(train_accuracy4)), train_accuracy4, 'g', label='FedAvg-layer_p=3')
    # plt.plot(range(len(train_accuracy5)), train_accuracy5, 'm', label='FedAvg-layer_p=4')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.legend()
    # plt.savefig('../save/all_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_acc_p.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs, args.unequal))

    # Plot Loss curve
    plt.figure()
    plt.title('MNIST IID E=3')
    plt.plot(range(len(global_acc1)), global_acc1, 'r', label='FedAvgLA p=0')
    plt.plot(range(len(global_acc2)), global_acc2, 'k', label='FedAvgLA p=1')
    plt.plot(range(len(global_acc3)), global_acc3, 'b', label='FedAvgLA p=2')
    plt.plot(range(len(global_acc4)), global_acc4, 'g', label='FedAvgLA p=3')
    plt.plot(range(len(global_acc5)), global_acc5, 'm', label='FedAvgLA p=4')
    ax = plt.gca()
    x_space = MultipleLocator(5)
    ax.xaxis.set_major_locator(x_space)
    ay = plt.gca()
    y_space = MultipleLocator(0.05)
    ay.yaxis.set_major_locator(y_space)
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Communication Rounds')
    plt.legend()
    plt.grid()
    plt.savefig('../save/alltest_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_unequal[{}]_acc_p.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.unequal))

    # fedavg = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[1]_B[{}]_unequal[{}]_p2.pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #             args.local_bs, args.unequal)
    # fedavglayer1 = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[2]_B[{}]_unequal[{}]_p2.pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #             args.local_bs, args.unequal)
    # fedavglayer2 = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[3]_B[{}]_unequal[{}]_p2.pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #             args.local_bs, args.unequal)
    # fedavglayer3 = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[4]_B[{}]_unequal[{}]_p2.pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #             args.local_bs, args.unequal)
    # fedavglayer4 = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[5]_B[{}]_unequal[{}]_p2.pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #             args.local_bs, args.unequal)
    # fedavglayer5 = '../save/objects/fedlayer_{}_{}_{}_C[{}]_iid[{}]_E[10]_B[{}]_unequal[{}]_p2.pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #             args.local_bs, args.unequal)
    #
    # with open(fedavg, 'rb') as f:
    #     train_loss1, train_accuracy1, global_loss1, global_acc1 = pickle.load(f)
    # with open(fedavglayer1, 'rb') as f:
    #     train_loss2, train_accuracy2, global_loss2, global_acc2 = pickle.load(f)
    # with open(fedavglayer2, 'rb') as f:
    #     train_loss3, train_accuracy3, global_loss3, global_acc3 = pickle.load(f)
    # with open(fedavglayer3, 'rb') as f:
    #     train_loss4, train_accuracy4, global_loss4, global_acc4 = pickle.load(f)
    # with open(fedavglayer4, 'rb') as f:
    #     train_loss5, train_accuracy5, global_loss5, global_acc5 = pickle.load(f)
    # with open(fedavglayer5, 'rb') as f:
    #     train_loss6, train_accuracy6, global_loss6, global_acc6 = pickle.load(f)
    #
    #
    # plt.figure()
    # plt.title('MNIST IID p=2')
    # plt.plot(range(len(global_acc1)), global_acc1, 'r', label='FedAvgLA E=1')
    # plt.plot(range(len(global_acc2)), global_acc2, 'k', label='FedAvgLA E=2')
    # plt.plot(range(len(global_acc3)), global_acc3, 'b', label='FedAvgLA E=3')
    # plt.plot(range(len(global_acc4)), global_acc4, 'g', label='FedAvgLA E=4')
    # plt.plot(range(len(global_acc5)), global_acc5, 'm', label='FedAvgLA E=5')
    # plt.plot(range(len(global_acc6)), global_acc6, 'c', label='FedAvgLA E=10')
    # ax = plt.gca()
    # x_space = MultipleLocator(10)
    # ax.xaxis.set_major_locator(x_space)
    # ay = plt.gca()
    # y_space = MultipleLocator(0.1)
    # ay.yaxis.set_major_locator(y_space)
    # plt.xlim([0, 100])
    # plt.ylim([0, 1])
    # plt.ylabel('Testing Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.legend()
    # plt.savefig('../save/alltest_{}_{}_{}_C[{}]_iid[{}]_E[X]_B[{}]_unequal[{}]_acc_E.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid,  args.local_bs, args.unequal))
