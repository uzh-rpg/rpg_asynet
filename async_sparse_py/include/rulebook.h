#pragma once
#include <iostream>
#include <vector>


class RuleBook
{
public:
    RuleBook(int H, int W, int k, int dimension)
    : k_(k), H_(H), W_(W), dimension_(dimension), inputs_(std::pow(k,dimension)), outputs_(std::pow(k,dimension))
    {
    }

    ~RuleBook() {};

    void addRule(int k, int in, int out)
    {
        outputs_[k].push_back(out);
        inputs_[k].push_back(in);
    };

    int nrules(int k)
    {
        return inputs_[k].size();
    };

    void getRules(int k, std::vector<int>& input, std::vector<int>& output)
    {
        input = inputs_[k];
        output = outputs_[k];
    };

    void print()
    {
        std::cout << "======> printing Rulebook"<< std::endl;
        std::string outstr = "";
        std::string instr = "";
        for (int i=0; i<std::pow(k_, dimension_); i++)
        {
            outstr += "{";
            instr += "{";

            for (int in : outputs_[i]) 
            {
                outstr += ("[" + std::to_string(in/W_) + "," + std::to_string(in%W_) + "] ");
            }
            
            for (int in : inputs_[i]) 
            {
                instr += ("[" + std::to_string(in/W_) + "," + std::to_string(in%W_) + "] ");
            }

            outstr += "}-";
            instr += "}-";
        }
        std::cout << "\toutputs: " << outstr << std::endl;
        std::cout << "\tinputs:  " << instr << std::endl;
        std::cout << "<====== "<< std::endl;
    }

    std::vector<std::vector<int>> inputs_;
    std::vector<std::vector<int>> outputs_;
    
    int k_;
    int H_;
    int W_;
    int dimension_;
};