% feyngraph56.mp -- do not edit, generated automatically by Physics513ClassNotes.tex
input feynmp
require_RCS_revision "1.30";
beginchar(1, 110*1.0pt#, 60*1.0pt#, 0);
"feynmf: 1";
LaTeX_unitlength:=1.0pt;
subgraph (0, 0, w, h);
vinit;
pickup pencircle scaled thin;
vleft (__i1, __i2);
vright (__o1, __o2);
vlabel ("$p_1$", __i1);
vlabel ("$p_2$", __i2);
vlabel ("$p_1'$", __o1);
vlabel ("$p_2'$", __o2);
vconnect ("fermion", __i1, __v1);
vconnect ("fermion", __v1, __i2);
vconnect ("fermion", __v2, __o1);
vconnect ("fermion", __o2, __v2);
vconnect ("dashes, label=$q$", __v1, __v2);
vfreeze;
vdraw;
endsubgraph;
endchar;
% the end.
end.
endinput;
