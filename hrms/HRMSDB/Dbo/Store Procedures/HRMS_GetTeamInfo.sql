CREATE PROCEDURE [dbo].[HRMS_GetAllTeamInfo.2.0.0]
	
AS
BEGIN
	SET NOCOUNT ON;
	
	SELECT 
		[TeamId],
		[DepartmentId], 
		d.[DepartmentName],
		[TeamName], 
		[StartTime], 
		[EndTime], 
		[TotalHour], 
		[IsEnable]
	FROM [dbo].[Team] WITH(NOLOCK)
	LEFT JOIN Department d WITH(NOLOCK) ON d.Id = [DepartmentId]

END
