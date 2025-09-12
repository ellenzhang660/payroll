import React, { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle, Clock, RefreshCw } from 'lucide-react';

// Types for our data structures
interface PayrollCheck {
  id: number;
  name: string;
  likelihood: number;
  status: 'passed' | 'failed' | 'pending';
}

interface PayrollRun {
  cycle: string;
  processedDate: string;
  overallLikelihood: number;
  checks: PayrollCheck[];
}

// Mock data - replace with actual API calls
const mockPayrollRun: PayrollRun = {
  cycle: '2025-07',
  processedDate: '28.08.2025, 17:05',
  overallLikelihood: 41,
  checks: [
    { id: 1, name: 'Check 1', likelihood: 53, status: 'passed' },
    { id: 2, name: 'Check 2', likelihood: 80, status: 'passed' },
    { id: 3, name: 'Check 3', likelihood: 6, status: 'failed' },
    { id: 4, name: 'Check 4', likelihood: 91, status: 'passed' },
    { id: 5, name: 'Check 5', likelihood: 35, status: 'failed' },
    { id: 6, name: 'Check 6', likelihood: 34, status: 'failed' },
    { id: 7, name: 'Check 7', likelihood: 8, status: 'failed' },
    { id: 8, name: 'Check 8', likelihood: 24, status: 'failed' },
  ]
};

// Individual check card component
const CheckCard: React.FC<{ check: PayrollCheck; onRunCheck: (id: number) => void }> = ({ 
  check, 
  onRunCheck 
}) => {
  const [isRunning, setIsRunning] = useState(false);

  const handleRunCheck = async () => {
    setIsRunning(true);
    await onRunCheck(check.id);
    setIsRunning(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'passed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      default: return 'bg-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed': return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'failed': return <AlertCircle className="w-4 h-4 text-red-600" />;
      default: return <Clock className="w-4 h-4 text-gray-600" />;
    }
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-900">{check.name}</h3>
        <div className="flex items-center gap-2">
          {getStatusIcon(check.status)}
          <span className="text-lg font-bold text-gray-900">{check.likelihood}%</span>
        </div>
      </div>
      
      <div className="mb-3">
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className={`h-2 rounded-full ${getStatusColor(check.status)}`}
            style={{ width: `${check.likelihood}%` }}
          ></div>
        </div>
      </div>
      
      <p className="text-sm text-gray-600 mb-3">
        Likelihood payroll passed this check.
      </p>
      
      <button
        onClick={handleRunCheck}
        disabled={isRunning}
        className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-md text-sm font-medium transition-colors"
      >
        {isRunning ? (
          <>
            <RefreshCw className="w-4 h-4 animate-spin" />
            Running...
          </>
        ) : (
          'Run Check'
        )}
      </button>
    </div>
  );
};

// Main dashboard component
const PayrollDashboard: React.FC = () => {
  const [payrollData, setPayrollData] = useState<PayrollRun>(mockPayrollRun);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Simulate API call for running individual check
  const runIndividualCheck = async (checkId: number): Promise<void> => {
    console.log(`Running check ${checkId}...`);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Update the specific check with new data
    setPayrollData(prev => ({
      ...prev,
      checks: prev.checks.map(check => 
        check.id === checkId 
          ? { 
              ...check, 
              likelihood: Math.floor(Math.random() * 100), 
              status: Math.random() > 0.5 ? 'passed' as const : 'failed' as const 
            }
          : check
      )
    }));
  };

  // Simulate API call for refreshing all data
  const refreshAllChecks = async (): Promise<void> => {
    setIsRefreshing(true);
    console.log('Refreshing all checks...');
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Update all checks
    const updatedChecks = payrollData.checks.map(check => ({
      ...check,
      likelihood: Math.floor(Math.random() * 100),
      status: Math.random() > 0.5 ? 'passed' as const : 'failed' as const
    }));
    
    const newOverallLikelihood = Math.floor(
      updatedChecks.reduce((sum, check) => sum + check.likelihood, 0) / updatedChecks.length
    );
    
    setPayrollData(prev => ({
      ...prev,
      checks: updatedChecks,
      overallLikelihood: newOverallLikelihood,
      processedDate: new Date().toLocaleString()
    }));
    
    setIsRefreshing(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-8 h-8 bg-green-500 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">P</span>
            </div>
            <h1 className="text-2xl font-bold text-gray-900">Payroll Ops</h1>
          </div>
          
          <div className="flex items-center gap-2 mb-4">
            <div className="w-4 h-4 bg-orange-400 rounded"></div>
            <span className="text-sm text-gray-600">
              AI-validated payroll — 20 automated checks
            </span>
          </div>
        </div>

        {/* Main Status Section */}
        <div className="bg-white rounded-lg border border-gray-200 p-6 mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Last Payroll Cycle Status</h2>
          <p className="text-gray-600 mb-6">
            This dashboard displays the most recent payroll run. An AI system evaluates{' '}
            <strong>20 checks</strong> and reports the <strong>likelihood of success</strong>{' '}
            for each one (0 - 100%).
          </p>
          
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-1">Payroll Run</h3>
              <p className="text-sm text-gray-500">
                Cycle: {payrollData.cycle} • Processed on {payrollData.processedDate}
              </p>
            </div>
            
            <div className="text-right">
              <p className="text-sm text-gray-500 mb-1">Overall likelihood of success</p>
              <p className="text-4xl font-bold text-gray-900">{payrollData.overallLikelihood}%</p>
            </div>
          </div>
          
          <div className="mt-6">
            <button
              onClick={refreshAllChecks}
              disabled={isRefreshing}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-green-400 text-white rounded-md font-medium transition-colors"
            >
              {isRefreshing ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Refreshing All Checks...
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4" />
                  Refresh All Checks
                </>
              )}
            </button>
          </div>
        </div>

        {/* AI Checks Grid */}
        <div className="mb-6">
          <h3 className="text-xl font-bold text-gray-900 mb-2">
            AI Checks (4 × 5)
          </h3>
          <p className="text-gray-600 mb-6">
            Each card represents one validation performed by the model.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {payrollData.checks.map((check) => (
            <CheckCard 
              key={check.id} 
              check={check} 
              onRunCheck={runIndividualCheck}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default PayrollDashboard;