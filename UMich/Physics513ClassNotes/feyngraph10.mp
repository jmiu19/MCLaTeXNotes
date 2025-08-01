% feyngraph10.mp -- do not edit, generated automatically by Physics513ClassNotes.tex
input feynmp
require_RCS_revision "1.30";
beginchar(1, 110*1.0pt#, 60*1.0pt#, 0);
"feynmf: 1";
LaTeX_unitlength:=1.0pt;
subgraph (0, 0, w, h);
vinit;
pickup pencircle scaled thin;
vleft (__i1, __i2);
vright (__o1);
vlabel ("$q_1$", __i1);
vlabel ("$q_2$", __i2);
vlabel ("$q_3$", __o1);
vconnect ("fermion", __i1, __v1);
vconnect ("fermion", __i2, __v1);
vconnect ("dashes_arrow", __v1, __o1);
vfreeze;
vdraw;
endsubgraph;
endchar;
% the end.
end.
endinput;
