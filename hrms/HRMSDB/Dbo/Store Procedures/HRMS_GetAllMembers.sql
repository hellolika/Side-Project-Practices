CREATE PROCEDURE [dbo].[HRMS_GetAllMembers.3.0.0]
	
AS
BEGIN
	SET NOCOUNT ON;
	
	SELECT m.[Id] AS [MemberId], 
		m.[Username],
		m.[Email],
		m.[Password],
		m.[PhoneNumber],
		m.[Address],
		m.[Salary],
		m.[Permission],
		m.[BankAccount],
		m.[IsInProbation],
		m.[IsResigned],
		m.[Remark],
		m.[TeamId],
		t.[TeamName],
		m.[JobGrade],
		m.[WorkLocationId],
		l.[LocationName] AS [WorkLocation],
		m.[IsSupervisor],
		m.[IsDeleted],
		m.[JoinDate],
        m.[IsAlertProbation],
		m.[BankName],
		m.[PositionId],
        m.[Position],
        m.[EmployeeId],
		m.[Gender],
		m.[IsFirstTimeUser],
		m.[Birthday],
		m.[NationalId],
		m.[VehicleType],
		m.[VehicleNumber],
		m.[DepartmentId],
		d.[DepartmentName],
		m.[IsManager],
		m.[IsCanSeeMemberSalary]
		FROM [dbo].[Member] m WITH(NOLOCK)
		LEFT JOIN [dbo].[Team] t WITH(NOLOCK) ON t.[TeamId] = m.[TeamId]
		LEFT JOIN [dbo].[LocationDetail] l WITH(NOLOCK) ON l.[Id] = m.[WorkLocationId]
		LEFT JOIN [dbo].[JobGrade] j WITH(NOLOCK) ON j.[Id] = m.[JobGrade]
		LEFT JOIN [dbo].[Department] d WITH(NOLOCK) ON d.[Id] = m.[DepartmentId]
		WHERE m.[IsDeleted] = 0
END
