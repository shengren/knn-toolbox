% MATLAB installation script

clear all
close all

tb_path = pwd;

tools = sprintf('%s/toolbox', tb_path);
lib = sprintf('%s/lib', tb_path);
bin = sprintf('%s/bin', tb_path);

p = path;
path(p, tools);
p = path;
path(p, lib);
p = path;
path(p, bin);
