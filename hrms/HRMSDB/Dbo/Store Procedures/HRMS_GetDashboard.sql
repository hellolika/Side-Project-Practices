CREATE PROCEDURE [dbo].[HRMS_GetDashboard.1.0.0]
	
AS
BEGIN
	DECLARE @allMemberCount as int
	DECLARE @allDepartmentsCount as int
	DECLARE @allLeaveRequest as int
	DECLARE @allEmployeesInProbation as int
	DECLARE @allPermanentEmployees as int
    DECLARE @totalLeaveToday as int

	SELECT @allMemberCount = COUNT([Id]) FROM [dbo].[Member] WHERE [IsDeleted] = 0 OR [IsDeleted] IS NULL
	SELECT @allDepartmentsCount = COUNT([TeamId]) FROM [dbo].[Team]
	SELECT @allLeaveRequest = COUNT([RequestId]) FROM [dbo].[TakeLeaveRecord] WHERE [Status] = 0 AND [IsCancel] = 0
		AND (MONTH([StartDate]) = MONTH(GETDATE()) AND YEAR([StartDate]) = YEAR(GETDATE())
			OR MONTH([EndDate]) = MONTH(GETDATE()) AND YEAR([EndDate]) = YEAR(GETDATE()));
	SELECT @totalLeaveToday = COUNT([RequestId]) FROM [dbo].[TakeLeaveRecord] WHERE [Status] = 1 AND [IsCancel] = 0 AND (CAST([StartDate] AS DATE) = CAST(DATEADD(hh, 7, GETDATE()) AS DATE) OR CAST(DATEADD(hh, 7, GETDATE()) AS DATE) BETWEEN [StartDate] AND [EndDate] OR CAST([EndDate] AS DATE) =CAST( DATEADD(hh, 7, GETDATE()) AS DATE));
	SELECT @allEmployeesInProbation = COUNT([Id]) FROM [dbo].[Member] WHERE [IsInProbation] = 1
		AND ([IsDeleted] = 0 OR [IsDeleted] IS NULL);
	SELECT @allPermanentEmployees = COUNT([Id]) FROM [dbo].[Member] WHERE [IsInProbation] = 0
		AND ([IsDeleted] = 0 OR [IsDeleted] IS NULL);

	SELECT @allMemberCount AS Employees, @allDepartmentsCount AS Departments, @allLeaveRequest AS LeaveRequests, @totalLeaveToday as TotalLeaveToday,
		   @allEmployeesInProbation AS ProbationEmployees, @allPermanentEmployees AS PermanentEmployees;

END